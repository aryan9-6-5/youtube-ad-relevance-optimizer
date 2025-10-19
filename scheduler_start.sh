# scheduler_start.sh
#!/bin/bash
set -e
echo "Waiting for Airflow database to be ready..."
for i in {1..30}; do
    if airflow db check; then
        echo "Database is ready!"
        exec airflow scheduler
    fi
    echo "Database not ready, retrying in 5 seconds... ($i/30)"
    sleep 5
done
echo "Failed to connect to database after 30 attempts"
exit 1