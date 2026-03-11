FROM golang:1.22-alpine AS builder
WORKDIR /app
COPY go.mod ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o /fraud-detection-api ./cmd/api

FROM alpine:3.19
WORKDIR /app
RUN adduser -D -g '' appuser
COPY --from=builder /fraud-detection-api .
USER appuser
EXPOSE 8080
CMD ["./fraud-detection-api"]
