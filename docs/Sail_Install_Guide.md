# Sail_Install_Guide<!-- omit in toc -->

以下提供两种方式分别在不同平台（PCIe、SoC）安装sail。
1. 源码编译安装
2. 通过预编译的whl包安装

更详细的Sail的安装可以参考[sophon-sail编译安装指南](https://doc.sophgo.com/sdk-docs/v23.07.01/docs_latest_release/docs/sophon-sail/docs/zh/html/1_build.html#)。

通常情况下，在运行环境通过源码编译sail不会有`python或者驱动版本依赖`问题，PCIe环境下推荐[源码编译安装](#源码编译安装)。Soc环境下python版本较为固定，建议使用[预编译的whl包安装](#预编译的whl包安装)。

此例程无需安装sophon-mw，源码编译时可设置`-DONLY_RUNTIME=ON`。

- [源码编译安装](#源码编译安装)
  - [x86/arm PCIe平台](#x86arm-pcie平台)
  - [SoC平台](#soc平台)
- [预编译的whl包安装](#预编译的whl包安装)

## 源码编译安装
### x86/arm PCIe平台

如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon，具体请参考[x86-pcie平台的开发和运行环境搭建](./Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](./Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

下载SOPHON-SAIL源码,解压后进入其源码目录，编译`不包含bmcv,sophon-ffmpeg,sophon-opencv`的SAIL, 通过此方式编译出来的SAIL无法使用其Decoder、Bmcv等多媒体相关接口。
```bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
python3 -m dfss --url=open@sophgo.com:sophon-demo/ChatGLM3/sail/sophon-sail_20240226.tar.gz
tar xvf sophon-sail_20240226.tar.gz
```

创建编译文件夹build,并进入build文件夹
```bash
cd sophon-sail
mkdir build && cd build
```
执行编译命令

```bash
cmake -DONLY_RUNTIME=ON ..
make pysail
```
打包生成python wheel,生成的wheel包的路径为‘python/pcie/dist’,文件名为‘sophon-3.7.0-py3-none-any.whl’
```bash
cd ../python/pcie
chmod +x sophon_pcie_whl.sh
./sophon_pcie_whl.sh
```
安装python wheel

```bash
pip3 install ./dist/sophon-3.7.0-py3-none-any.whl --force-reinstall
```

### SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。

`使用指定版本的python3(和目标SOC上的python3保持一致)`,通过交叉编译的方式,编译出`不包含bmcv,sophon-ffmpeg,sophon-opencv的SAIL`, python3的安装方式可通过python官方网站获取, 也可以根据[获取在X86主机上进行交叉编译的Python3]获取已经编译好的python3。 本示例使用的python3路径为‘python_3.8.2/bin/python3’,python3的动态库目录‘python_3.8.2/lib’。

如果您需要其他版本的sophon-sail，可以参考上一小节，下载源码自己编译，参考[sail交叉编译方法](https://doc.sophgo.com/sdk-docs/v23.07.01/docs_latest_release/docs/sophon-sail/docs/zh/html/1_build.html#id5)。

下载SOPHON-SAIL源码,解压后进入其源码目录。通过此方式编译出来的SAIL无法使用其Decoder、Bmcv等多媒体相关接口。

创建编译文件夹build,并进入build文件夹
```bash
mkdir build && cd build
```
执行编译命令
```bash
cmake -DBUILD_TYPE=soc  \
    -DONLY_RUNTIME=ON \
    -DCMAKE_TOOLCHAIN_FILE=../cmake/BM168x_SOC/ToolChain_aarch64_linux.cmake \
    -DPYTHON_EXECUTABLE=python_3.8.2/bin/python3 \
    -DCUSTOM_PY_LIBDIR=python_3.8.2/lib \
    -DLIBSOPHON_BASIC_PATH=libsophon_soc_0.4.1_aarch64/opt/sophon/libsophon-0.4.1 ..
make pysail
```
打包生成python wheel,生成的wheel包的路径为‘python/soc/dist’,文件名为‘sophon_arm-3.7.0-py3-none-any.whl’
```bash
cd ../python/soc
chmod +x sophon_soc_whl.sh
./sophon_soc_whl.sh
```
安装python wheel

将‘sophon_arm-3.7.0-py3-none-any.whl’拷贝到目标SOC上,然后执行如下安装命令
```bash
pip3 install sophon_arm-3.7.0-py3-none-any.whl --force-reinstall
```

## 预编译的whl包安装

目前大多设备python版本为3.8.2，可直接下载此whl包

```bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
python3 -m dfss --url=open@sophgo.com:sophon-demo/ChatGLM3/sail/soc/sophon_arm-3.7.0-py3-none-any.whl
```

下面提供了在不同版本python、libsophon、sophon-mw下预编译好的whl包，用户可根据自己的机器环境选择安装。可通过`ls /opt/sophon`查看运行环境各sdk版本
```bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
python3 -m dfss --url=open@sophgo.com:ezoo/chatdoc/python_wheels.zip
unzip python_wheels.zip
```

文件目录如下
```
python_wheels
├── arm_pcie
│   ├── libsophon-0.4.4_sophonmw-0.5.1
│   │   ├── py310
│   │   │   └── sophon_arm_pcie-3.7.0-py3-none-any.whl
│   │   ├── py35
│   │   │   └── sophon_arm_pcie-3.7.0-py3-none-any.whl
│   │   ├── py36
│   │   │   └── sophon_arm_pcie-3.7.0-py3-none-any.whl
│   │   ├── py37
│   │   │   └── sophon_arm_pcie-3.7.0-py3-none-any.whl
│   │   ├── py38
│   │   │   └── sophon_arm_pcie-3.7.0-py3-none-any.whl
│   │   └── py39
│   │       └── sophon_arm_pcie-3.7.0-py3-none-any.whl
│   ├── libsophon-0.4.6_sophonmw-0.6.0
│   │   ├── py310
│   │   │   └── sophon_arm_pcie-3.7.0-py3-none-any.whl
│   │   ├── py35
│   │   │   └── sophon_arm_pcie-3.7.0-py3-none-any.whl
│   │   ├── py36
│   │   │   └── sophon_arm_pcie-3.7.0-py3-none-any.whl
│   │   ├── py37
│   │   │   └── sophon_arm_pcie-3.7.0-py3-none-any.whl
│   │   ├── py38
│   │   │   └── sophon_arm_pcie-3.7.0-py3-none-any.whl
│   │   └── py39
│   │       └── sophon_arm_pcie-3.7.0-py3-none-any.whl
│   ├── libsophon-0.4.8_sophonmw-0.6.2
│   │   ├── py310
│   │   │   └── sophon_arm_pcie-3.7.0-py3-none-any.whl
│   │   ├── py35
│   │   │   └── sophon_arm_pcie-3.7.0-py3-none-any.whl
│   │   ├── py36
│   │   │   └── sophon_arm_pcie-3.7.0-py3-none-any.whl
│   │   ├── py37
│   │   │   └── sophon_arm_pcie-3.7.0-py3-none-any.whl
│   │   ├── py38
│   │   │   └── sophon_arm_pcie-3.7.0-py3-none-any.whl
│   │   └── py39
│   │       └── sophon_arm_pcie-3.7.0-py3-none-any.whl
│   └── libsophon-0.4.9_sophonmw-0.7.0
│       ├── py310
│       │   └── sophon_arm_pcie-3.7.0-py3-none-any.whl
│       ├── py35
│       │   └── sophon_arm_pcie-3.7.0-py3-none-any.whl
│       ├── py36
│       │   └── sophon_arm_pcie-3.7.0-py3-none-any.whl
│       ├── py37
│       │   └── sophon_arm_pcie-3.7.0-py3-none-any.whl
│       ├── py38
│       │   └── sophon_arm_pcie-3.7.0-py3-none-any.whl
│       └── py39
│           └── sophon_arm_pcie-3.7.0-py3-none-any.whl
├── loongarch
│   ├── libsophon-0.4.8_runtime
│   │   ├── py310
│   │   │   └── sophon_loongarch64-3.7.0-py3-none-any.whl
│   │   ├── py37
│   │   │   └── sophon_loongarch64-3.7.0-py3-none-any.whl
│   │   ├── py38
│   │   │   └── sophon_loongarch64-3.7.0-py3-none-any.whl
│   │   └── py39
│   │       └── sophon_loongarch64-3.7.0-py3-none-any.whl
│   └── libsophon-0.4.9_sophonmw-0.7.0
│       ├── py310
│       │   └── sophon_loongarch64-3.7.0-py3-none-any.whl
│       ├── py37
│       │   └── sophon_loongarch64-3.7.0-py3-none-any.whl
│       ├── py38
│       │   └── sophon_loongarch64-3.7.0-py3-none-any.whl
│       └── py39
│           └── sophon_loongarch64-3.7.0-py3-none-any.whl
├── soc_BM1684_BM1684X
│   ├── libsophon-0.4.4_sophonmw-0.5.1
│   │   ├── py310
│   │   │   └── sophon_arm-3.7.0-py3-none-any.whl
│   │   └── py38
│   │       └── sophon_arm-3.7.0-py3-none-any.whl
│   ├── libsophon-0.4.6_sophonmw-0.6.0
│   │   ├── py310
│   │   │   └── sophon_arm-3.7.0-py3-none-any.whl
│   │   └── py38
│   │       └── sophon_arm-3.7.0-py3-none-any.whl
│   ├── libsophon-0.4.8_sophonmw-0.6.2
│   │   ├── py310
│   │   │   └── sophon_arm-3.7.0-py3-none-any.whl
│   │   └── py38
│   │       └── sophon_arm-3.7.0-py3-none-any.whl
│   └── libsophon-0.4.9_sophonmw-0.7.0
│       ├── py310
│       │   └── sophon_arm-3.7.0-py3-none-any.whl
│       └── py38
│           └── sophon_arm-3.7.0-py3-none-any.whl
└── soc_BM1688
    └── libsophon-0.4.9_sophonmw-1.2.0
        ├── py310
        │   └── sophon_arm-3.7.0-py3-none-any.whl
        └── py38
            └── sophon_arm-3.7.0-py3-none-any.whl
```

选择适合运行环境的版本安装，例如
```bash
pip3 install sophon_arm-3.7.0-py3-none-any.whl --force-reinstall
```
