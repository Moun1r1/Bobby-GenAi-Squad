# Bobby Studio frontend — Next.js + tRPC. Talks only to the Python backend (BACKEND_URL) via tRPC procedures.
FROM node:20-alpine

WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1

COPY package.json ./
RUN npm install

COPY . .
RUN npm run build

EXPOSE 3000
CMD ["npm", "run", "start"]
