# ----------------------
# 第 1 阶段：构建阶段
# ----------------------
    FROM node:20-alpine AS builder

    WORKDIR /app
    
    # 复制 package.json 和 lock 文件
    COPY package*.json ./
    COPY .env ./.env
    # 安装依赖
    RUN npm install
    
    # 复制所有项目文件
    COPY . .
    
    # 构建生产版本
    RUN npm run build
    
    
    # ----------------------
    # 第 2 阶段：生产运行阶段
    # ----------------------
    FROM node:20-alpine AS runner
    
    ENV NODE_ENV=production
    
    WORKDIR /app
    
    # 拷贝构建产物和运行所需文件
    COPY --from=builder /app/package.json ./
    COPY --from=builder /app/package-lock.json ./
    COPY --from=builder /app/.next ./.next
    COPY --from=builder /app/public ./public
    COPY --from=builder /app/next.config.* ./
    COPY --from=builder /app/node_modules ./node_modules
    
    # 可选：如果你使用 src/ 目录作为页面入口
    COPY --from=builder /app/src ./src
    
    # 可选：复制配置文件
    COPY --from=builder /app/drizzle.config.ts ./
    COPY --from=builder /app/.env .env
    
    EXPOSE 3000
    
    CMD ["npm", "start"]
    