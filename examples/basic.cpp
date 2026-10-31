#include <minecraft_finders/geode.hpp>
#include <iostream>

int main() {
    geode::Options options;
    geode::set_version(options, "26.3");
    options.radius = 512;
    options.prefilter = 8;
    options.threads = 1;
    options.scan_backend = "cpu";
    auto result = geode::search(options);
    std::cout << result.centers.size() << '\n';
    return result.stats.scanned > 0 ? 0 : 1;
}
