#!/bin/bash
# cloud_setup/setup_gcp.sh

PROJECT_ID=$1

echo "Creating GCP project: $PROJECT_ID"
gcloud projects create $PROJECT_ID

echo "Setting project"
gcloud config set project $PROJECT_ID

echo "Enabling services"
gcloud services enable compute.googleapis.com storage.googleapis.com

echo "✅ GCP setup complete!"
