#include "coarse_gpu.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace geode {
namespace {
using Handle = void *;
using UInt = uint32_t;
using ULong = uint64_t;

struct API {
    HMODULE library = LoadLibraryExW(L"OpenCL.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
#define DECLARE(name, result, ...)                                                                                     \
    using name##Fn = result(__stdcall *)(__VA_ARGS__);                                                                 \
    name##Fn name{};
    DECLARE(clGetPlatformIDs, int, UInt, Handle *, UInt *)
    DECLARE(clGetDeviceIDs, int, Handle, ULong, UInt, Handle *, UInt *)
    DECLARE(clGetDeviceInfo, int, Handle, UInt, size_t, void *, size_t *)
    DECLARE(clCreateContext, Handle, void *, UInt, Handle *, void *, void *, int *)
    DECLARE(clCreateCommandQueue, Handle, Handle, Handle, ULong, int *)
    DECLARE(clCreateProgramWithSource, Handle, Handle, UInt, const char **, const size_t *, int *)
    DECLARE(clBuildProgram, int, Handle, UInt, Handle *, const char *, void *, void *)
    DECLARE(clGetProgramBuildInfo, int, Handle, Handle, UInt, size_t, void *, size_t *)
    DECLARE(clCreateKernel, Handle, Handle, const char *, int *)
    DECLARE(clCreateBuffer, Handle, Handle, ULong, size_t, void *, int *)
    DECLARE(clSetKernelArg, int, Handle, UInt, size_t, const void *)
    DECLARE(clEnqueueNDRangeKernel, int, Handle, Handle, UInt, const size_t *, const size_t *, const size_t *, UInt,
            Handle *, Handle *)
    DECLARE(clEnqueueReadBuffer, int, Handle, Handle, UInt, size_t, size_t, void *, UInt, Handle *, Handle *)
    DECLARE(clEnqueueWriteBuffer, int, Handle, Handle, UInt, size_t, size_t, const void *, UInt, Handle *, Handle *)
    DECLARE(clReleaseMemObject, int, Handle)
    DECLARE(clReleaseKernel, int, Handle)
    DECLARE(clReleaseProgram, int, Handle)
    DECLARE(clReleaseCommandQueue, int, Handle)
    DECLARE(clReleaseContext, int, Handle)
#undef DECLARE

    bool load() {
        if (!library)
            return false;
#define LOAD(name)                                                                                                     \
    name = std::bit_cast<name##Fn>(GetProcAddress(library, #name));                                                    \
    if (!name)                                                                                                         \
        return false;
        LOAD(clGetPlatformIDs)
        LOAD(clGetDeviceIDs)
        LOAD(clGetDeviceInfo)
        LOAD(clCreateContext)
        LOAD(clCreateCommandQueue)
        LOAD(clCreateProgramWithSource)
        LOAD(clBuildProgram)
        LOAD(clGetProgramBuildInfo)
        LOAD(clCreateKernel)
        LOAD(clCreateBuffer)
        LOAD(clSetKernelArg)
        LOAD(clEnqueueNDRangeKernel)
        LOAD(clEnqueueReadBuffer)
        LOAD(clEnqueueWriteBuffer)
        LOAD(clReleaseMemObject)
        LOAD(clReleaseKernel)
        LOAD(clReleaseProgram)
        LOAD(clReleaseCommandQueue)
        LOAD(clReleaseContext)
#undef LOAD
        return true;
    }

    ~API() {
        if (library)
            FreeLibrary(library);
    }
};

void check(int code) {
    if (code)
        throw std::runtime_error("OpenCL coarse scan failed: " + std::to_string(code));
}

struct Session {
    API &api;
    Handle context{}, queue{}, program{}, place{}, count{}, grid{}, points{}, counter{};

    ~Session() {
        for (auto b : {grid, points, counter})
            if (b)
                api.clReleaseMemObject(b);
        for (auto k : {place, count})
            if (k)
                api.clReleaseKernel(k);
        if (program)
            api.clReleaseProgram(program);
        if (queue)
            api.clReleaseCommandQueue(queue);
        if (context)
            api.clReleaseContext(context);
    }
};

constexpr const char *kernel = R"CLC(
    ulong mix(ulong v) {
        v = (v ^ (v >> 30)) * 0xBF58476D1CE4E5B9UL;
        v = (v ^ (v >> 27)) * 0x94D049BB133111EBUL;
        return v ^ (v >> 31);
    }

    __kernel void place(__global uint *grid, int width, int height, int ox, int oz, ulong seed, ulong xs, ulong zs) {
        int x = get_global_id(0), z = get_global_id(1);
        if (x >= width || z >= height)
            return;
        int cx = ox + x, cz = oz + z;
        uint bits = 0xffffffffU;
        if (cx >= -1875000 && cx <= 1874999 && cz >= -1875000 && cz <= 1874999) {
            ulong v = (((ulong)(long)cx * xs + (ulong)(long)cz * zs) ^ seed) + 20002UL;
            v ^= 0x6A09E667F3BCC909UL;
            ulong a = mix(v), b = mix(v + 0x9E3779B97F4A7C15UL);
            bits = (uint)((rotate(a + b, (ulong)17) + a) >> 40);
        }
        grid[z * width + x] = (bits < 699051U) + ((bits < 699050U) << 16);
    }

    __kernel void count(__global uint *grid, __global uint2 *output, __global uint *counter, int width,
                        int source_width, int ox, int oz, int maxx, int maxz, int threshold, int centerx, int centerz,
                        int radius, int circle) {
        __local uint prefix[SIDE * SIDE];
        int lid = get_local_id(0), gx = get_group_id(0) * 16, gz = get_group_id(1) * 16;
        for (int i = lid; i < SIDE * SIDE; i += 256)
            prefix[i] = grid[(gz + i / SIDE) * source_width + gx + i % SIDE];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < SIDE) {
            uint sum = 0;
            for (int x = 0; x < SIDE; ++x) {
                sum += prefix[lid * SIDE + x];
                prefix[lid * SIDE + x] = sum;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
#if MODERN
        // SIDE不超过80；两组16位前缀计数均小于65536，不会跨组进位。
        if (lid < SIDE) {
            uint sum = 0;
            for (int z = 0; z < SIDE; ++z) {
                sum += prefix[z * SIDE + lid];
                prefix[z * SIDE + lid] = sum;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        int x = lid % 16, z = lid / 16, cx = ox + gx + x, cz = oz + gz + z;
        if (cx > maxx || cz > maxz)
            return;
        if (circle) {
            long dx = (long)clamp(centerx, cx * 16, cx * 16 + 15) - centerx,
                 dz = (long)clamp(centerz, cz * 16, cz * 16 + 15) - centerz;
            if (dx * dx + dz * dz > (long)radius * radius)
                return;
        }
        uint sum = 0;
#if MODERN
        int right = x + 2 * Q, bottom = z + 2 * Q;
        sum = prefix[bottom * SIDE + right];
        if (x)
            sum -= prefix[bottom * SIDE + x - 1];
        if (z)
            sum -= prefix[(z - 1) * SIDE + right];
        if (x && z)
            sum += prefix[(z - 1) * SIDE + x - 1];
#else
        for (int dz = -Q; dz <= Q; ++dz) {
            int2 span = spans[dz + Q];
            if (span.y < span.x)
                continue;
            int row = (z + dz + Q) * SIDE, left = x + Q + span.x, right = x + Q + span.y;
            sum += prefix[row + right] - (left ? prefix[row + left - 1] : 0);
        }
#endif
        if ((sum & 65535U) >= (uint)threshold) {
            uint slot = atomic_inc(counter);
            output[slot] = (uint2)((gz + z) * width + gx + x, sum);
        }
    }
)CLC";
}

bool gpu_scan(const Options &o, int minx, int maxx, int minz, int maxz, const std::function<int()> &threshold,
              const std::function<void(int, int, int, int)> &offer,
              const std::function<void(uint64_t, uint64_t)> &progress, const std::function<bool(uint64_t)> &completed,
              const std::function<void(uint64_t, uint64_t)> &commit, std::string *failure_reason,
              const std::function<void(const std::string &)> &on_device) {
    auto fail = [&](const char *why) {
        if (failure_reason) *failure_reason = why;
        return false;
    };
    API api;
    if (!api.load())
        return fail("no-opencl");
    UInt size = 0;
    int status = api.clGetPlatformIDs(0, nullptr, &size);
    if (status == -1001 || !size)
        return fail("no-opencl");
    check(status);
    std::vector<Handle> platforms(size);
    check(api.clGetPlatformIDs(size, platforms.data(), nullptr));
    int q = o.activity_radius(), side = 16 + 2 * q;
    // Try every platform/device pair: hybrid laptops often list the iGPU first,
    // and a device whose local memory is too small must not force a CPU fallback
    // while another GPU could run the kernel.
    Handle device = nullptr;
    ULong local_memory = 0;
    size_t group_size = 0;
    char name_buffer[256] = {};
    for (auto p : platforms) {
        UInt count = 0;
        if (api.clGetDeviceIDs(p, 4, 0, nullptr, &count) != 0 || count == 0)
            continue;
        std::vector<Handle> devices(count);
        if (api.clGetDeviceIDs(p, 4, count, devices.data(), nullptr) != 0)
            continue;
        for (auto candidate : devices) {
            ULong memory = 0;
            size_t group = 0;
            if (api.clGetDeviceInfo(candidate, 0x1023, sizeof(memory), &memory, nullptr) != 0)
                continue;
            if (api.clGetDeviceInfo(candidate, 0x1004, sizeof(group), &group, nullptr) != 0)
                continue;
            if (group < 256 || uint64_t(side) * side * 4 > memory)
                continue;
            device = candidate;
            local_memory = memory;
            group_size = group;
            api.clGetDeviceInfo(candidate, 0x102B, sizeof(name_buffer), name_buffer, nullptr);
            break;
        }
        if (device)
            break;
    }
    if (!device)
        return fail("no-device");
    std::ostringstream code;
    code << "#define MODERN " << int(o.modern) << "\n#define Q " << q << "\n#define SIDE " << side
         << "\n__constant int2 spans[" << 2 * q + 1 << "]={";
    for (int dz = -q; dz <= q; ++dz) {
        int lo = -q, hi = q;
        if (!o.modern) {
            int distance = dz > 0 ? dz * 16 - 7 : dz < 0 ? -dz * 16 - 8 : 0;
            int64_t remaining = int64_t(o.tick_radius) * o.tick_radius - 1 - int64_t(distance) * distance;
            if (remaining < 0) {
                lo = 0;
                hi = -1;
            } else {
                int reach = int(std::sqrt(double(remaining)));
                while (int64_t(reach + 1) * (reach + 1) <= remaining)
                    ++reach;
                while (int64_t(reach) * reach > remaining)
                    --reach;
                lo = -((reach + 8) / 16);
                hi = (reach + 7) / 16;
            }
        }
        if (dz != -q)
            code << ',';
        code << "(int2)(" << lo << ',' << hi << ')';
    }
    code << "};\n" << kernel;
    auto source = code.str();
    const char *raw = source.c_str();
    int error = 0;
    Session s{api};
    s.context = api.clCreateContext(nullptr, 1, &device, nullptr, nullptr, &error);
    check(error);
    s.queue = api.clCreateCommandQueue(s.context, device, 0, &error);
    check(error);
    s.program = api.clCreateProgramWithSource(s.context, 1, &raw, nullptr, &error);
    check(error);
    status = api.clBuildProgram(s.program, 1, &device, "-cl-std=CL1.2", nullptr, nullptr);
    if (status) {
        size_t length = 0;
        check(api.clGetProgramBuildInfo(s.program, device, 0x1183, 0, nullptr, &length));
        std::vector<char> log(length + 1);
        check(api.clGetProgramBuildInfo(s.program, device, 0x1183, length, log.data(), nullptr));
        throw std::runtime_error("OpenCL build failed: " + std::string(log.data()));
    }
    s.place = api.clCreateKernel(s.program, "place", &error);
    check(error);
    s.count = api.clCreateKernel(s.program, "count", &error);
    check(error);
    // The device is selected and the kernel is built: from here the scan really
    // runs on the GPU, so report it before the (long) batch loop starts.
    if (on_device) on_device(std::string(name_buffer));
    constexpr int batch = 4096;

    struct Point {
        UInt index, counts;
    };

    static_assert(sizeof(Point) == 8);
    s.grid = api.clCreateBuffer(s.context, 1, size_t(batch + 2 * q) * (batch + 2 * q) * 4, nullptr, &error);
    check(error);
    s.points = api.clCreateBuffer(s.context, 1, size_t(batch) * batch * sizeof(Point), nullptr, &error);
    check(error);
    s.counter = api.clCreateBuffer(s.context, 1, 4, nullptr, &error);
    check(error);
    auto arg = [&](Handle k, UInt id, const auto &value) {
        check(api.clSetKernelArg(k, id, sizeof(value), &value));
    };
    Random random(o.seed);
    ULong xs = (random.java_long() | 1) * 16, zs = (random.java_long() | 1) * 16;
    uint64_t total = uint64_t((maxx - minx + batch) / batch) * ((maxz - minz + batch) / batch), done = 0;
    std::vector<Point> points;
    for (int az = minz; az <= maxz; az += batch)
        for (int ax = minx; ax <= maxx; ax += batch) {
            int bx = std::min(maxx, ax + batch - 1), bz = std::min(maxz, az + batch - 1);
            if (o.checkpoint)
                o.checkpoint();
            if (completed && completed(done)) {
                ++done;
                continue;
            }
            if (o.contains(std::clamp(o.center_x, ax * 16, bx * 16 + 15),
                           std::clamp(o.center_z, az * 16, bz * 16 + 15))) {
                int width = (bx - ax + 16) / 16 * 16, height = (bz - az + 16) / 16 * 16, sw = width + 2 * q,
                    sh = height + 2 * q, ox = ax - q, oz = az - q;
                arg(s.place, 0, s.grid);
                arg(s.place, 1, sw);
                arg(s.place, 2, sh);
                arg(s.place, 3, ox);
                arg(s.place, 4, oz);
                arg(s.place, 5, o.seed);
                arg(s.place, 6, xs);
                arg(s.place, 7, zs);
                UInt count = 0;
                check(api.clEnqueueWriteBuffer(s.queue, s.counter, 1, 0, 4, &count, 0, nullptr, nullptr));
                size_t global[2] = {size_t((sw + 15) / 16 * 16), size_t((sh + 15) / 16 * 16)}, local[2] = {16, 16};
                check(api.clEnqueueNDRangeKernel(s.queue, s.place, 2, nullptr, global, local, 0, nullptr, nullptr));
                int cut = threshold(), circle = o.shape == "circle";
                arg(s.count, 0, s.grid);
                arg(s.count, 1, s.points);
                arg(s.count, 2, s.counter);
                arg(s.count, 3, width);
                arg(s.count, 4, sw);
                arg(s.count, 5, ax);
                arg(s.count, 6, az);
                arg(s.count, 7, bx);
                arg(s.count, 8, bz);
                arg(s.count, 9, cut);
                arg(s.count, 10, o.center_x);
                arg(s.count, 11, o.center_z);
                arg(s.count, 12, o.radius);
                arg(s.count, 13, circle);
                global[0] = size_t(width / 16) * 256;
                global[1] = size_t(height / 16);
                local[0] = 256;
                local[1] = 1;
                check(api.clEnqueueNDRangeKernel(s.queue, s.count, 2, nullptr, global, local, 0, nullptr, nullptr));
                check(api.clEnqueueReadBuffer(s.queue, s.counter, 1, 0, 4, &count, 0, nullptr, nullptr));
                points.resize(count);
                if (count)
                    check(api.clEnqueueReadBuffer(s.queue, s.points, 1, 0, points.size() * sizeof(Point), points.data(),
                                                  0, nullptr, nullptr));
                size_t offered = 0;
                for (auto p : points) {
                    if ((offered++ % 1024) == 0 && o.checkpoint)
                        o.checkpoint();
                    offer(ax + int(p.index % width), az + int(p.index / width), int(p.counts & 65535),
                          int(p.counts >> 16));
                }
            }
            if (commit)
                commit(done, uint64_t(bx - ax + 1) * (bz - az + 1));
            if (progress)
                progress(++done, total);
        }
    return true;
}
}
#else
namespace geode {
bool gpu_scan(const Options &, int, int, int, int, const std::function<int()> &,
              const std::function<void(int, int, int, int)> &, const std::function<void(uint64_t, uint64_t)> &,
              const std::function<bool(uint64_t)> &, const std::function<void(uint64_t, uint64_t)> &,
              std::string *) {
    return false;
}
}
#endif
