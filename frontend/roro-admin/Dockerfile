
    FROM node:20-alpine AS builder

    WORKDIR /roro
        

    COPY package*.json ./
    COPY . .       
        
    RUN npm install && npm run build

    FROM node:20-alpine AS runner
    
    ENV NODE_ENV=production
    WORKDIR /roro
        

    COPY --from=builder /roro/package.json .
    COPY --from=builder /roro/node_modules ./node_modules
    COPY --from=builder /roro/.next ./.next
    COPY --from=builder /roro/public ./public
    COPY --from=builder /roro/next.config.ts .
    COPY --from=builder /roro/src ./src
    COPY --from=builder /roro/package-lock.json .
    COPY --from=builder /roro/drizzle.config.ts .
    
    EXPOSE 3001

    CMD ["npx","next","start","-p","3001"]