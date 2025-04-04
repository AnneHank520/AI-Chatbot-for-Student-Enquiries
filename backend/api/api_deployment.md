
# API部署指南

## 1. 服务器需求

对于支持60个并发用户的部署，建议以下配置：
- **CPU**: 4核
- **内存**: 8GB
- **存储**: 最少30GB SSD
- **操作系统**: Ubuntu 20.04/22.04

## 2. 环境设置

### 安装基础软件包

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv nginx supervisor
```

### 创建项目目录

```bash
mkdir -p /var/www/docassist
cd /var/www/docassist
```

### 设置Python虚拟环境

```bash
python3 -m venv venv
source venv/bin/activate
```

### 克隆项目

```bash
git clone <您的项目仓库URL> .
```

### 安装依赖

```bash
pip install -r backend/api/requirements.txt
pip install gunicorn
```

## 3. 配置文件

### 创建.env文件

```bash
cd /var/www/docassist/backend/api
touch .env
echo "DASHSCOPE_API_KEY=您的API密钥" > .env
```

### 创建Gunicorn配置

创建文件 `/var/www/docassist/gunicorn_config.py`:

```python
bind = "127.0.0.1:8000"
workers = 4
timeout = 120
worker_class = "sync"
```

### 创建Supervisor配置

创建文件 `/etc/supervisor/conf.d/docassist.conf`:

```ini
[program:docassist]
directory=/var/www/docassist
command=/var/www/docassist/venv/bin/gunicorn -c gunicorn_config.py backend.api.api:app
user=www-data
autostart=true
autorestart=true
stderr_logfile=/var/log/docassist/api.err.log
stdout_logfile=/var/log/docassist/api.out.log
```

创建日志目录：

```bash
sudo mkdir -p /var/log/docassist
sudo chown -R www-data:www-data /var/log/docassist
```

### 配置Nginx

创建文件 `/etc/nginx/sites-available/docassist`:

```nginx
server {
    listen 80;
    server_name your_domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static {
        alias /var/www/docassist/static;
    }

    # 增加上传文件大小限制
    client_max_body_size 20M;
}
```

启用站点：

```bash
sudo ln -s /etc/nginx/sites-available/docassist /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## 4. 创建上传和模型目录

```bash
mkdir -p /var/www/docassist/backend/api/uploads
mkdir -p /var/www/docassist/backend/api/models
sudo chown -R www-data:www-data /var/www/docassist/backend/api/uploads
sudo chown -R www-data:www-data /var/www/docassist/backend/api/models
```

## 5. 启动应用

```bash
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start docassist
```

## 6. SSL配置（推荐）

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your_domain.com
```

## 7. 监控与日志

检查日志文件以排查问题：

```bash
tail -f /var/log/docassist/api.out.log
tail -f /var/log/docassist/api.err.log
```

## 8. 性能优化建议

### 预先加载模型

确保首次启动前预先加载模型：

```bash
cd /var/www/docassist
source venv/bin/activate
python -c "from backend.api.api import load_or_create_model; load_or_create_model()"
```

### 配置缓存

若需处理大量相似查询，考虑添加Redis缓存：

```bash
sudo apt install -y redis-server
pip install redis
```

在`api.py`中实现缓存可提高响应速度。

### 文件存储优化

对于生产环境，考虑将上传文件存储在对象存储服务（如AWS S3）中，而不是本地文件系统。

## 9. 扩展建议

如果60个并发用户不足以满足需求，可以考虑：

1. 增加Gunicorn workers数量（每个worker需要额外内存）
2. 设置多台服务器并使用负载均衡器
3. 将向量数据库迁移到专用服务器

## 10. 故障排除

常见问题：

- **500错误**: 检查API日志文件
- **模型加载问题**: 确保有足够内存和正确的文件权限
- **上传失败**: 检查目录权限和Nginx上传大小限制
- **API密钥问题**: 确保.env文件存在且权限正确

## 11. 安全建议

- 设置防火墙仅开放必要端口
- 定期更新系统和依赖包
- 实施请求限流以防止DOS攻击
- 考虑添加API认证机制

## 12. 维护流程

- 定期备份上传的文件和向量数据库
- 监控磁盘空间使用情况
- 设置自动系统更新

