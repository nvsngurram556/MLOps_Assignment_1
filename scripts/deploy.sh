#!/bin/bash

# Deployment script for California Housing ML API
# Usage: ./scripts/deploy.sh [local|staging|production]

set -e

ENVIRONMENT=${1:-local}
PROJECT_NAME="california-housing-ml-api"

echo "🚀 Deploying $PROJECT_NAME to $ENVIRONMENT environment..."

case $ENVIRONMENT in
    local)
        echo "📦 Building and starting services locally..."
        
        # Build and start services
        docker-compose down --remove-orphans
        docker-compose build
        docker-compose up -d
        
        echo "⏳ Waiting for services to start..."
        sleep 30
        
        # Health checks
        echo "🔍 Running health checks..."
        
        # Check API health
        if curl -f http://localhost:8000/health; then
            echo "✅ ML API is healthy"
        else
            echo "❌ ML API health check failed"
            exit 1
        fi
        
        # Check MLflow
        if curl -f http://localhost:5000; then
            echo "✅ MLflow is running"
        else
            echo "⚠️ MLflow might not be ready yet"
        fi
        
        # Check Prometheus
        if curl -f http://localhost:9090; then
            echo "✅ Prometheus is running"
        else
            echo "⚠️ Prometheus might not be ready yet"
        fi
        
        # Check Grafana
        if curl -f http://localhost:3000; then
            echo "✅ Grafana is running"
        else
            echo "⚠️ Grafana might not be ready yet"
        fi
        
        echo ""
        echo "🎉 Local deployment completed successfully!"
        echo ""
        echo "📊 Access the services:"
        echo "  • ML API: http://localhost:8000"
        echo "  • API Docs: http://localhost:8000/docs"
        echo "  • MLflow UI: http://localhost:5000"
        echo "  • Prometheus: http://localhost:9090"
        echo "  • Grafana: http://localhost:3000 (admin/admin)"
        echo ""
        echo "🧪 Test the API:"
        echo 'curl -X POST "http://localhost:8000/predict" \'
        echo '     -H "Content-Type: application/json" \'
        echo '     -d '"'"'{"features": {"MedInc": 8.3014, "HouseAge": 21.0, "AveRooms": 6.238137, "AveBedrms": 0.971880, "Population": 2401.0, "AveOccup": 2.109842, "Latitude": 37.86, "Longitude": -122.22}}'"'"
        ;;
    
    staging)
        echo "🏗️ Deploying to staging environment..."
        
        # Build and tag image
        docker build -t $PROJECT_NAME:staging .
        
        # Push to registry (uncomment and configure as needed)
        # docker tag $PROJECT_NAME:staging your-registry/$PROJECT_NAME:staging
        # docker push your-registry/$PROJECT_NAME:staging
        
        # Deploy to staging server (example using docker run)
        # ssh staging-server "docker pull your-registry/$PROJECT_NAME:staging && docker stop $PROJECT_NAME || true && docker run -d --name $PROJECT_NAME -p 8000:8000 your-registry/$PROJECT_NAME:staging"
        
        echo "⚠️ Staging deployment logic needs to be configured for your infrastructure"
        ;;
    
    production)
        echo "🔥 Deploying to production environment..."
        
        # Add production deployment logic
        # This would typically involve:
        # 1. Building and tagging the production image
        # 2. Running security scans
        # 3. Deploying to production infrastructure (K8s, ECS, etc.)
        # 4. Running smoke tests
        # 5. Monitoring deployment
        
        echo "⚠️ Production deployment logic needs to be configured for your infrastructure"
        ;;
    
    *)
        echo "❌ Unknown environment: $ENVIRONMENT"
        echo "Usage: $0 [local|staging|production]"
        exit 1
        ;;
esac