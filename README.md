# 创建 conda 环境

1. 安装 anaconda 或 miniconda （linux）
```bash
# 下载
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
  && rm Miniconda3-latest-Linux-x86_64.sh 
# 激活
source /opt/conda/bin/activate
# 初始化从conda环境
conda init --all 
```

2. 创建虚拟环境
```bash
conda create -n train python=3.9 # 名字train + python 版本3.9
```

3. 激活虚拟环境
```bash
conda activate train # 激活虚拟环境
conda deactivate # 退出当前虚拟环境到base
conda env list # 查看现有虚拟环境
```
4. 设置 pip 路径 (若发现无法直接使用pip，将软件安装如虚拟环境中请输入如下代码)
```bash
alias pip='虚拟环境的位置/bin/pip'
alias pip='/opt/conda/envs/train/bin/pip'
```

5. 安装 torch
```bash
pip install torch==2.3.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```

6. 配置需要的包
```bash
wget -O requirements.txt https://www.pan.hekaiyu.com.cn/d/file/python/env/requirements.txt?sign=SmPNzV5IjN2sZL9p4iDz675aiVM7M4ASiB7JNoq2ceo=:0
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

7. 预先下载好的包
```bash
# flash attention——link: https://github.com/Dao-AILab/flash-attention/releases
wget -O flash_attn.whl https://www.pan.hekaiyu.com.cn/d/file/python/env/flash_attn-2.5.9.post1%2Bcu122torch2.3cxx11abiFALSE-cp39-cp39-linux_x86_64.whl?sign=vLhTjFIATZySJXRgC-DKy1-QPsClehE3wwCL37FCTLs=:0
pip install flash_attn.whl

# deepspeed——link: https://github.com/deepspeedai/DeepSpeed/releases
wget -O deepspeed.whl https://www.pan.hekaiyu.com.cn/d/file/python/env/deepspeed-0.14.2%2Bcu121torch2.3-cp39-cp39-manylinux_2_24_x86_64.whl?sign=hye9A04XMevsNhCRCE1lBMDrsu-jVeWSDg6GvgjDIJc=:0
pip install deepspeed.whl
```


# Github使用指南（待修改）
1. 将 Git 与 Github 绑定

- 获取 ssh key
```Git
cd ~/.ssh
```
- 若返回 "no such file or directory" 表明电脑没有ssh key，创建ssh key
```Git
ssh-keygen -t rsa -C “git账号邮箱”
```

- 在 .shh 目录下获取 id_rsa.pub 文件里面存储的是公钥并绑定到到自己的 GitHub 上
![ssh.png](image%2Fssh.png)

- 在 Git bash 中，输入： 
```
ssh -T git@github.com 
```
检查是否绑定成功

- 配置绑定信息
```Git
git config --global user.name “gitname”
git config --global user.email “git邮箱”
```

2. 提交代码
```Git
cd /path/to/your/project # 进入本地项目目录
git remote add origin https://github.com/Kaiyu-He/env.git # 关联远程仓库
git init # 初始化 git 仓库
git add . # 提交代码
git commit -m "Local changes"
git push --force origin main # 推送代码
```


5. 拉取项目
```Git
git pull origin main
```

5. 获取 github 项目的代码
```git
git clone https://github.com/Kaiyu-He/env.git
```