# NEMWAS Docker Deployment Guide

This directory contains Docker configuration for deploying NEMWAS with full monitoring stack.

## Quick Start

```bash
# From the project root directory
make docker-up

# Or manually:
docker-compose -f docker/docker-compose.yml up -d
```

## Services

The docker-compose.yml creates the following services:

### 1. NEMWAS Core (`nemwas`)
- Main NEMWAS application
- API on port 8080
- Metrics on port 9090
- Privileged mode for NPU access

### 2. Prometheus (`prometheus`)
- Metrics collection
- Web UI on port 9091
- Configured to scrape NEMWAS metrics

### 3. Grafana (`grafana`)
- Visualization dashboards
- Web UI on port 3000
- Default login: admin/admin

### 4. Redis (`redis`)
- Optional caching layer
- Port 6379

## Accessing Services

After starting the stack:

- **NEMWAS API**: http://localhost:8080
- **API Docs**: http://localhost:8080/docs
- **Prometheus**: http://localhost:9091
- **Grafana**: http://localhost:3000 (admin/admin)
- **Redis**: localhost:6379

## NPU Support in Docker

For NPU support, the container needs:

1. **Privileged mode** (already configured)
2. **Device access** to `/dev/dri` (already configured)
3. **NPU drivers** installed on the host

If you have Intel Core Ultra with NPU:
```bash
# On the host machine
./scripts/setup_npu.sh

# Then restart Docker containers
docker-compose -f docker/docker-compose.yml restart
```

## Configuration

### Environment Variables

You can override configuration using environment variables:

```bash
# Create .env file
cat > docker/.env << EOF
NEMWAS_LOG_LEVEL=DEBUG
NEMWAS_MAX_AGENTS=100
NEMWAS_API_PORT=8080
EOF

# Start with env file
docker-compose --env-file docker/.env up -d
```

### Custom Configuration

To use a custom configuration:

1. Create your config file: `config/my-config.yaml`
2. Update docker-compose.yml:
   ```yaml
   volumes:
     - ./config/my-config.yaml:/app/config/production.yaml:ro
   ```

## Monitoring

### Grafana Dashboards

1. Access Grafana at http://localhost:3000
2. Login with admin/admin
3. Go to Dashboards → Browse
4. Import NEMWAS dashboard

### Prometheus Queries

Example queries for Prometheus:

```promql
# Total tasks processed
sum(nemwas_tasks_total)

# Task success rate
rate(nemwas_tasks_total{status="success"}[5m]) / rate(nemwas_tasks_total[5m])

# Average task duration
histogram_quantile(0.95, nemwas_task_duration_seconds_bucket)

# NPU utilization
nemwas_npu_utilization_percent

# Active agents
nemwas_active_agents
```

## Volumes

Data is persisted in Docker volumes:

- `nemwas-models`: Model files
- `nemwas-data`: Application data
- `prometheus-data`: Metrics history
- `grafana-data`: Dashboard configurations
- `redis-data`: Cache data

### Backup

```bash
# Backup all volumes
docker run --rm -v nemwas-models:/models -v backup:/backup alpine tar czf /backup/models.tar.gz -C /models .
docker run --rm -v nemwas-data:/data -v backup:/backup alpine tar czf /backup/data.tar.gz -C /data .
```

### Restore

```bash
# Restore from backup
docker run --rm -v nemwas-models:/models -v backup:/backup alpine tar xzf /backup/models.tar.gz -C /models
docker run --rm -v nemwas-data:/data -v backup:/backup alpine tar xzf /backup/data.tar.gz -C /data
```

## Scaling

To run multiple NEMWAS instances:

```yaml
# In docker-compose.yml
services:
  nemwas:
    deploy:
      replicas: 3
```

Or manually:
```bash
docker-compose -f docker/docker-compose.yml up -d --scale nemwas=3
```

## Troubleshooting

### Check logs
```bash
# All services
docker-compose -f docker/docker-compose.yml logs -f

# Specific service
docker-compose -f docker/docker-compose.yml logs -f nemwas
```

### Enter container
```bash
docker exec -it nemwas-core bash
```

### Test NPU inside container
```bash
docker exec -it nemwas-core python -c "import openvino as ov; print(ov.Core().available_devices)"
```

### Resource usage
```bash
docker stats nemwas-core
```

## Production Deployment

For production deployment:

1. **Use secrets** for sensitive data:
   ```yaml
   secrets:
     api_key:
       file: ./secrets/api_key.txt
   ```

2. **Enable SSL/TLS**:
   - Use a reverse proxy (nginx/traefik)
   - Configure proper certificates

3. **Set resource limits** appropriately:
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '8'
         memory: 16G
   ```

4. **Configure logging**:
   ```yaml
   logging:
     driver: "json-file"
     options:
       max-size: "100m"
       max-file: "10"
   ```

5. **Use external volumes** for data persistence
