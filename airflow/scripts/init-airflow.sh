#!/bin/bash

# Airflow Initialization Script for ConformAI
# This script initializes Airflow database and creates admin user

set -e

echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║            ConformAI Airflow Initialization Script               ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Error: docker-compose is not installed"
    echo "Please install docker-compose first: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✓ Docker Compose found"
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found"
    echo "Creating .env from .env.example..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "✓ Created .env file"
        echo "⚠️  Please edit .env and add your API keys before continuing"
        echo ""
        read -p "Press Enter when you're ready to continue..."
    else
        echo "❌ Error: .env.example not found"
        echo "Please create a .env file with required configuration"
        exit 1
    fi
fi

echo "✓ Environment file found"
echo ""

# Check required environment variables
echo "Checking required environment variables..."

required_vars=(
    "ANTHROPIC_API_KEY"
    "OPENAI_API_KEY"
    "POSTGRES_USER"
    "POSTGRES_PASSWORD"
    "POSTGRES_DB"
)

missing_vars=()

for var in "${required_vars[@]}"; do
    if ! grep -q "^${var}=" .env 2>/dev/null || grep "^${var}=$" .env &>/dev/null; then
        missing_vars+=("$var")
    fi
done

if [ ${#missing_vars[@]} -ne 0 ]; then
    echo "❌ Error: The following required environment variables are missing or empty:"
    for var in "${missing_vars[@]}"; do
        echo "   - $var"
    done
    echo ""
    echo "Please add them to your .env file"
    exit 1
fi

echo "✓ All required environment variables are set"
echo ""

# Generate Fernet key if not exists
if ! grep -q "^AIRFLOW__CORE__FERNET_KEY=" .env || grep "^AIRFLOW__CORE__FERNET_KEY=$" .env &>/dev/null; then
    echo "Generating Airflow Fernet key..."
    FERNET_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
    echo "AIRFLOW__CORE__FERNET_KEY=${FERNET_KEY}" >> .env
    echo "✓ Fernet key generated and added to .env"
    echo ""
fi

# Generate webserver secret key if not exists
if ! grep -q "^AIRFLOW__WEBSERVER__SECRET_KEY=" .env || grep "^AIRFLOW__WEBSERVER__SECRET_KEY=$" .env &>/dev/null; then
    echo "Generating Airflow webserver secret key..."
    SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    echo "AIRFLOW__WEBSERVER__SECRET_KEY=${SECRET_KEY}" >> .env
    echo "✓ Webserver secret key generated and added to .env"
    echo ""
fi

# Start required services (PostgreSQL and Redis)
echo "Starting prerequisite services (PostgreSQL, Redis)..."
docker-compose up -d postgres redis

echo "Waiting for PostgreSQL to be ready..."
sleep 5

# Wait for PostgreSQL to be healthy
max_attempts=30
attempt=1
while [ $attempt -le $max_attempts ]; do
    if docker-compose exec -T postgres pg_isready -U conformai &>/dev/null; then
        echo "✓ PostgreSQL is ready"
        break
    fi
    echo "  Waiting for PostgreSQL... (attempt $attempt/$max_attempts)"
    sleep 2
    attempt=$((attempt + 1))
done

if [ $attempt -gt $max_attempts ]; then
    echo "❌ Error: PostgreSQL failed to start"
    exit 1
fi

echo ""

# Initialize Airflow database
echo "Initializing Airflow database..."
docker-compose run --rm airflow-webserver airflow db init

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to initialize Airflow database"
    exit 1
fi

echo "✓ Airflow database initialized"
echo ""

# Create admin user
echo "Creating Airflow admin user..."
echo ""

# Check if admin user already exists
if docker-compose run --rm airflow-webserver airflow users list 2>/dev/null | grep -q "admin"; then
    echo "⚠️  Admin user already exists"
    echo ""
    read -p "Do you want to reset the admin password? (y/N): " reset_password

    if [ "$reset_password" = "y" ] || [ "$reset_password" = "Y" ]; then
        read -sp "Enter new admin password: " admin_password
        echo ""
        read -sp "Confirm admin password: " admin_password_confirm
        echo ""

        if [ "$admin_password" != "$admin_password_confirm" ]; then
            echo "❌ Error: Passwords don't match"
            exit 1
        fi

        # Delete existing user and recreate
        docker-compose run --rm airflow-webserver airflow users delete -u admin
        docker-compose run --rm airflow-webserver airflow users create \
            --username admin \
            --firstname Admin \
            --lastname User \
            --role Admin \
            --email admin@conformai.com \
            --password "$admin_password"

        echo "✓ Admin password updated"
    fi
else
    # Prompt for admin password
    read -sp "Enter admin password (default: admin): " admin_password
    echo ""

    if [ -z "$admin_password" ]; then
        admin_password="admin"
    fi

    docker-compose run --rm airflow-webserver airflow users create \
        --username admin \
        --firstname Admin \
        --lastname User \
        --role Admin \
        --email admin@conformai.com \
        --password "$admin_password"

    echo "✓ Admin user created"
fi

echo ""

# Start all Airflow services
echo "Starting Airflow services..."
docker-compose up -d airflow-webserver airflow-scheduler airflow-worker

echo ""
echo "Waiting for Airflow webserver to be ready..."
sleep 5

max_attempts=30
attempt=1
while [ $attempt -le $max_attempts ]; do
    if curl -s http://localhost:8080/health &>/dev/null; then
        echo "✓ Airflow webserver is ready"
        break
    fi
    echo "  Waiting for webserver... (attempt $attempt/$max_attempts)"
    sleep 2
    attempt=$((attempt + 1))
done

if [ $attempt -gt $max_attempts ]; then
    echo "⚠️  Warning: Airflow webserver health check timed out"
    echo "The webserver might still be starting up. Check logs with:"
    echo "  docker-compose logs -f airflow-webserver"
fi

echo ""
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║                Airflow Initialization Complete!                  ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""
echo "🌐 Airflow UI: http://localhost:8080"
echo "👤 Username: admin"
echo "🔑 Password: [the password you just set]"
echo ""
echo "📋 Next steps:"
echo "  1. Open http://localhost:8080 in your browser"
echo "  2. Login with admin credentials"
echo "  3. Find the 'eu_legal_documents_pipeline' DAG"
echo "  4. Toggle it ON to enable automatic execution"
echo "  5. Click 'Trigger DAG' to run it manually"
echo ""
echo "📊 Monitor services:"
echo "  docker-compose ps"
echo "  docker-compose logs -f airflow-scheduler"
echo "  docker-compose logs -f airflow-worker"
echo ""
echo "📖 Documentation: ./airflow/README.md"
echo ""
