# 

1. 安装 anaconda 或 miniconda
2. 创建虚拟环境
```shell
conda create
```

# Github使用指南（待修）
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