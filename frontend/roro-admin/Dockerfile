# ----------------------
# 第 1 阶段：构建阶段
# ----------------------
    FROM node:20-alpine AS builder

    WORKDIR /roro
        
    # 只复制必要的文件（移除 .env 相关操作）
    COPY package*.json ./
    COPY . .       
        
    RUN npm install && npm run build
    
    # ----------------------
    # 第 2 阶段：生产环境
    # ----------------------
    FROM node:20-alpine AS runner
    
    ENV NODE_ENV=production
    WORKDIR /roro
        
    # 只复制必要的运行时文件（移除 .env 相关操作）
    COPY --from=builder /roro/package.json .
    COPY --from=builder /roro/node_modules ./node_modules
    COPY --from=builder /roro/.next ./.next
    COPY --from=builder /roro/public ./public
    COPY --from=builder /roro/next.config.ts .
    COPY --from=builder /roro/src ./src
    COPY --from=builder /roro/package-lock.json .
    COPY --from=builder /roro/drizzle.config.ts .
    
    EXPOSE 3000
    CMD ["npm", "start"]