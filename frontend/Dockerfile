FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

# We run a build and start for production
RUN npm run build
EXPOSE 3000

CMD ["npm", "start"]
