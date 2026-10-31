# 紫水晶母岩引擎

## 功能与边界

独立计算晶洞、局部地形影响和母岩热点。支持游戏随机刻范围与自定义水平圆/方形，结果为模型预计值；有限候选预算可能漏热点。支持1.18至26.3，GPU仅用于可选OpenCL粗筛，精排使用CPU。地形模型在有限预算下启用两段式精确裁剪：先用纯几何模型算出母岩数上界，只对仍可能进榜的候选跑地形模拟，并自检回退，最终榜单与单遍一致；详见使用说明。

## 构建

需要CMake、Ninja与C++20编译器，Windows使用MinGW。已有build目录沿用缓存生成器。普通构建不需要Java、Python、tests或开发工具。

```powershell
.\build.ps1
# 仅完整本地开发环境使用：
.\build.ps1 -Test
```

生成`build/geode-finder.exe`。本引擎独立构建，不依赖相邻仓库。

## 运行

在本项目根目录执行：

```powershell
.\run.bat
.\run.bat --seed 12345 --radius 10000 --limit 20
.\build\geode-finder.exe --config finder.json --output results
.\build\geode-finder.exe --help
```

`run.bat`读取本目录存在的finder.json，再追加命令行覆盖值；程序本身只在明确传入`--config`时读取配置，不会暗中查找文件。所有引擎使用`--limit`控制结果条数，默认100；`--output`始终为目录，默认results，由后端生成CSV/TXT/JSON并返回实际文件路径。

## 配置

统一使用UTF-8 JSON。键名就是命令行参数名去掉前面的`--`，不是桌面JSONL报文；种子必须写为字符串。

```json
{
  "seed": "0",
  "version": "26.3",
  "radius": 10000,
  "shape": "circle",
  "threads": 8,
  "limit": 100,
  "output": "results"
}
```

优先级为**命令行覆盖值 → JSON配置 → 引擎内置默认值**。相对路径以启动工作目录为准；run.bat会先切到本项目目录。配置开关使用true/false。格/区块由后端换算，外半径用radius与radius-unit；算法专属参数见[使用说明](docs/说明/使用说明.md)。finder.json保留本项目已有默认搜索偏好，示例中的版本不代表所有项目文件都被改成同一版本。

## 桌面接口

三个引擎统一使用`--desktop`，通过stdin/stdout交换协议v3的UTF-8 JSON Lines。finder为`geode`。桌面直接发送完整任务配置，不经过run.bat、不读取finder.json；长期设置与进程调度由桌面壳负责，正式计数、排名和文件内容由C++负责。配置文件与JSONL请求分别用于独立CLI和桌面调用，不互相套用。

## 作为C++算法库使用

公开入口是`<minecraft_finders/geode.hpp>`，目标为`MinecraftFinders::Geode`。将完整源码仓库作为子目录加入你的CMake项目即可；这里是源码集成接口，不承诺跨编译器二进制ABI。

```cmake
add_subdirectory(path/to/geode-native geode-build)
target_link_libraries(my_program PRIVATE MinecraftFinders::Geode)
```

直接调用示例见[examples/basic.cpp](examples/basic.cpp)。嵌入其他项目时默认只构建算法库；独立构建默认保留CLI和桌面协议。`GEODE_BUILD_CLI`可显式切换，`GEODE_BUILD_EXAMPLE=ON`可构建示例；`BUILD_TESTING`默认关闭。不需要tests、tools、Java或相邻Cubiomes仓库，也不要求程序经由JSON或子进程调用算法。

本引擎选择Cubiomes模块：**Biomes + Java263**。`third_party/cubiomes/`是固定提交的最小源码快照，锁文件和revision文件一并保留；升级时从集中维护仓库导出所需模块，将ZIP内容放入`third_party/`，整体替换旧快照，勿在副本中修库。各引擎可以锁定不同已验收提交。

发布构建启用函数/数据分节、链接裁剪和符号剥离，保留`-O3`与原有浮点顺序；`GEODE_ENABLE_IPO`控制可用时的Release跨文件优化。Windows静态链接编译器运行库，不需要Cubiomes DLL。不使用fast-math、UPX或本机专属指令集。

## 目录与说明

`include/`为公开入口，`examples/`为直接调用示例，`src/`为算法和入口，`third_party/`为固定依赖，`build/`为本地构建，`results/`为CLI输出，`docs/说明/`为现行说明。已有的data目录保存必要编译常量。tests、tools和docs/记录仅本地保留，不参与普通源码构建，也不作为桌面运行依赖。

[使用说明](docs/说明/使用说明.md) · [第三方来源](THIRD_PARTY.md)


Cubiomes使用[集中维护分支](https://github.com/qk-yiyihehe/cubiomes/tree/minecraft-finders)的固定源码子集；[依赖锁文件](third_party/cubiomes-lock.json)记录提交、补丁和SHA256。普通构建不联网，结果以`cubiomesCommit`回显依赖身份；升级不等于原版精度已验证。

## 许可证

自有代码采用GNU GPL第3版（GPL-3.0-only），完整条款见[LICENSE](LICENSE)。第三方代码保留各自版权和许可，详见[THIRD_PARTY.md](THIRD_PARTY.md)。
