
    FROM node:20-alpine AS builder

    WORKDIR /app
    

    COPY package*.json ./
    COPY .env ./.env

    RUN npm install
    

    COPY . .
    

    RUN npm run build
    
    

    FROM node:20-alpine AS runner
    
    ENV NODE_ENV=production
    
    WORKDIR /app
    

    COPY --from=builder /app/package.json ./
    COPY --from=builder /app/package-lock.json ./
    COPY --from=builder /app/.next ./.next
    COPY --from=builder /app/public ./public
    COPY --from=builder /app/next.config.* ./
    COPY --from=builder /app/node_modules ./node_modules
    COPY --from=builder /app/src ./src
    COPY --from=builder /app/drizzle.config.ts ./
    COPY --from=builder /app/.env .env
    
    EXPOSE 3000
    
    CMD ["npm", "start"]
    
