# 第三方来源与许可

本仓库自有代码采用[GPL-3.0-only](LICENSE)。以下外部代码与参考材料保留原作者版权及原许可证；项目许可证不改写这些声明。

## 外部开源项目

| 来源 | 使用范围 | 版本与许可 |
|---|---|---|
| [Cubiomes维护分支](https://github.com/qk-yiyihehe/cubiomes/tree/minecraft-finders)，源于Cubitect/cubiomes与xpple/cubiomes | `third_party/cubiomes/`为所需C源码及本地include闭包；噪声、RNG、群系及调用到的结构能力 | 固定维护提交`ea575d85536eba8e89a4e2a0528a60d2094221c8`，基于xpple提交`aee74239609f3b371747e4573808c04a72518979`；[锁文件](third_party/cubiomes-lock.json)含文件SHA256、来源与补丁；[MIT原文](third_party/cubiomes/LICENSE) |
| [JSON for Modern C++](https://github.com/nlohmann/json)，Niels Lohmann及贡献者 | `third_party/json.hpp`，用于配置、JSON Lines通信及结果序列化 | `v3.12.0`；[MIT原文](third_party/json-LICENSE.MIT)，头文件内其他原始声明保留 |
| [geode-finder](https://github.com/soulsoiledit/geode-finder)，soiledit | 晶洞模型的早期算法参考；现代活动范围的CPU滑动窗口粗筛参考其`src/search.rs` | 粗筛参考提交`70f5171e3ef1090f8bc02ab27c458f56080dab3c`；[MIT原文](third_party/geode-finder-LICENSE.MIT) |

负Y限制使用新版上游实现，Java纵向量化epsilon修复已移入集中维护分支，禁止再在本副本单独修一遍。本引擎的晶洞、结构拼接、地形、含水层和雕刻仍由自有代码实现；更新依赖不会自动替换这些算法或26.3自有float实现。

## 游戏行为与数据参照

[Minecraft Java Edition](https://www.minecraft.net/)由Mojang Studios/Microsoft开发，是晶洞、局部地形、结构生成及版本差异的行为参照。开发工具可读取本地游戏文件，生成原生数值算子、系数和几何常量；冻结回归保留坐标及判定结果。

仓库不分发Minecraft JAR、官方数据包、NBT模板或反编译缓存。游戏及官方资源的权利归原权利人，不能以本仓库GPL声明替代其授权。开发期Java参照工具不参与正式运行。

## 分发时保留的文件

分发本项目时保留[LICENSE](LICENSE)、本说明、相关MIT许可证及源码中的版权声明。分发二进制时还需按GPL第3版条款提供相应源代码。OpenCL运行时使用系统驱动，本仓库不分发显卡驱动或GPU SDK。

26.3通用float噪声的C++适配位于依赖中的`extensions/java263_noise.hpp`，由维护库统一维护；它保留原产品的[GPL-3.0-only许可](third_party/cubiomes/extensions/LICENSE)，未改为上游MIT。产品src不再保留同名实现。搜索、布局和本引擎的版本模型仍属于产品代码。

26.3结构气候改为直接调用Cubiomes公共采样器；原产品气候求值器与对应表已移除。母岩专用密度图、含水层、晶洞生成及搜索仍在产品内维护。C++适配中的Perlin运算转调维护库MIT核心，适配文件本身保留原GPL许可。
