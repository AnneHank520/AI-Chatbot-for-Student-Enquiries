# ----------------------
# 第 1 阶段：构建阶段
# ----------------------
    FROM node:20-alpine AS builder

    # 设置工作目录
    WORKDIR /roro
    
    # 复制 package.json 和 package-lock.json
    COPY package*.json ./
    
    # 安装依赖
    RUN npm install
    
    # 复制项目所有文件到容器
    COPY . .
    
    # 运行构建（Next.js默认的构建命令）
    RUN npm run build
    
    # ----------------------
    # 第 2 阶段：生产环境
    # ----------------------
    FROM node:20-alpine AS runner
    
    # 设置环境变量
    ENV NODE_ENV=production
    
    # 设置工作目录
    WORKDIR /roro
    
    # 只复制需要的文件到 runner 镜像，保持最终镜像精简
    COPY --from=builder /roro/package.json .
    COPY --from=builder /roro/node_modules ./node_modules
    COPY --from=builder /roro/.next ./.next
    COPY --from=builder /roro/public ./public
    COPY --from=builder /roro/next.config.ts .
    COPY --from=builder /roro/.env ./.env
    COPY --from=builder /roro/src ./src
    COPY --from=builder /roro/package-lock.json .
    COPY --from=builder /roro/drizzle.config.ts .
    
    # 暴露容器的 3000 端口（Next.js 默认端口）
    EXPOSE 3000
    
    # 启动命令
    CMD ["npm", "start"]
    