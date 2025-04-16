# ----------------------
# Phase 1: build phase
# ----------------------
    FROM node:20-alpine AS builder

    WORKDIR /app
    
    # Copy the package.json and lock files
    COPY package*.json ./
    COPY .env ./.env
    # Installation of dependencies
    RUN npm install
    
    # Copy all project files
    COPY . .
    
    # Building the production version
    RUN npm run build
    
    
    # ----------------------
    # Phase 2: Production run phase
    # ----------------------
    FROM node:20-alpine AS runner
    
    ENV NODE_ENV=production
    
    WORKDIR /app
    
    # Copy the build product and the files needed to run it
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
    
