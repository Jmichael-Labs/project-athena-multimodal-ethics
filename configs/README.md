# Configuration Files for Project Athena

This directory contains comprehensive configuration files for different deployment environments and scenarios.

## Configuration Files Overview

### Environment-Specific Configurations

#### `development.yaml`
- **Purpose**: Local development and debugging
- **Characteristics**: 
  - Smaller models for faster loading
  - Lower thresholds for testing edge cases
  - Debug logging enabled
  - Hot reload and profiling enabled
- **Usage**: `python -m athena --config configs/development.yaml`

#### `staging.yaml`
- **Purpose**: Pre-production testing and validation
- **Characteristics**:
  - Production-like settings with relaxed constraints
  - Performance testing enabled
  - Comprehensive audit logging
  - Load testing capabilities
- **Usage**: `python -m athena --config configs/staging.yaml`

#### `production.yaml`
- **Purpose**: Production deployment with enterprise features
- **Characteristics**:
  - Largest models for maximum accuracy
  - Strict ethics thresholds
  - High availability and auto-scaling
  - Comprehensive security and compliance
- **Usage**: `python -m athena --config configs/production.yaml`

#### `testing.yaml`
- **Purpose**: Automated testing and CI/CD pipelines
- **Characteristics**:
  - Mock models and external APIs
  - Deterministic behavior for reproducible tests
  - Fast execution with minimal resource usage
  - Comprehensive test data and fixtures
- **Usage**: `pytest --config configs/testing.yaml`

### Deployment Configurations

#### `docker-compose.yaml`
Complete Docker stack for local or single-machine deployment including:
- Athena Ethics Engine
- PostgreSQL database
- Redis cache
- Prometheus monitoring
- Grafana dashboards
- Elasticsearch + Kibana for logs
- Nginx load balancer

**Usage**: `docker-compose up -d`

#### `kubernetes.yaml`
Production-ready Kubernetes manifests for scalable cloud deployment including:
- Horizontal Pod Autoscaling
- Persistent storage for models and data
- Service mesh configuration
- Network policies for security
- Ingress with SSL termination

**Usage**: `kubectl apply -f configs/kubernetes.yaml`

### Special Configuration Files

#### `constitution_production.yaml`
Comprehensive Constitutional AI principles and guidelines including:
- Core ethical principles (helpfulness, harmlessness, honesty, etc.)
- Contextual guidelines for different scenarios
- Decision frameworks and conflict resolution
- Implementation guidelines with confidence thresholds
- Monitoring and adaptation protocols

## Configuration Structure

### Core Components

All configuration files follow a consistent structure:

```yaml
# Modality configurations
text:
  enabled: true
  model_name: "meta-llama/Llama-2-70b-chat-hf"
  threshold: 0.85
  batch_size: 32
  preprocessing: {}
  postprocessing: {}

image:
  enabled: true
  model_name: "meta-cv/clip-vit-large-patch14"
  # ... similar structure

audio:
  enabled: true
  model_name: "meta-audio/wav2vec2-large-960h"
  # ... similar structure

video:
  enabled: true
  model_name: "meta-video/videomae-large"
  # ... similar structure

# Meta AI integration
meta_integration:
  llama_models: ["llama-70b", "llama-13b"]
  api_endpoint: "https://api.meta.ai/v1"
  rate_limit: 10000
  # ... additional settings

# Ethics thresholds
ethics_thresholds:
  harmful_content: 0.95
  bias_detection: 0.85
  privacy_violation: 0.98
  # ... all threshold categories

# Framework configurations
rlhf:
  enabled: true
  # ... RLHF settings

constitutional_ai:
  enabled: true
  # ... Constitutional AI settings

# General settings
logging_level: "INFO"
cache_enabled: true
monitoring_enabled: true
# ... additional settings
```

### Key Configuration Sections

#### Modality Settings
- **enabled**: Whether the modality is active
- **model_name**: Specific model to use (Meta models preferred)
- **threshold**: Confidence threshold for flagging content
- **batch_size**: Processing batch size for performance
- **preprocessing**: Input processing parameters
- **postprocessing**: Output formatting options

#### Ethics Thresholds
Confidence thresholds for different ethical categories:
- `harmful_content`: General harmful content detection
- `bias_detection`: Bias and discrimination detection
- `privacy_violation`: Privacy and data protection
- `misinformation`: False or misleading information
- `toxicity`: Toxic language and behavior
- `hate_speech`: Hate speech and harassment
- `violence`: Violent content and threats
- `sexual_content`: Sexual or adult content
- `child_safety`: Child protection (highest priority)
- `copyright_infringement`: Intellectual property violations

#### Meta Integration
- **llama_models**: List of available Llama models
- **api_endpoint**: Meta AI API endpoint
- **rate_limit**: API request rate limit
- **use_pytorch_integration**: Use PyTorch-based integration
- **enable_fair_research_mode**: Enable Meta FAIR research features

## Environment Variables

The following environment variables can override configuration settings:

### Required Variables
- `META_API_KEY`: Meta AI API authentication key
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string

### Optional Variables
- `ATHENA_CONFIG_ENV`: Environment name (development, staging, production, testing)
- `ATHENA_LOG_LEVEL`: Override logging level
- `ATHENA_DEBUG`: Enable debug mode
- `ATHENA_CACHE_TTL`: Cache time-to-live in seconds

### Security Variables
- `ENCRYPTION_KEY`: Master encryption key for sensitive data
- `JWT_SECRET`: Secret for JWT token signing
- `SSL_CERT_PATH`: Path to SSL certificate
- `SSL_KEY_PATH`: Path to SSL private key

## Usage Examples

### Development Setup
```bash
# Set environment
export ATHENA_CONFIG_ENV=development
export ATHENA_DEBUG=true

# Run with development config
python -m athena.main --config configs/development.yaml
```

### Production Deployment
```bash
# Set production environment variables
export META_API_KEY=your_meta_api_key
export DATABASE_URL=postgresql://user:pass@host:5432/athena
export REDIS_URL=redis://host:6379/0

# Deploy with Docker Compose
docker-compose -f configs/docker-compose.yaml up -d

# Or deploy to Kubernetes
kubectl apply -f configs/kubernetes.yaml
```

### Testing
```bash
# Run tests with testing config
pytest --config configs/testing.yaml tests/

# Run specific test suite
python -m pytest tests/test_ethics_engine.py --config configs/testing.yaml
```

## Configuration Validation

All configurations include validation to ensure:
- Threshold values are between 0 and 1
- Port numbers are in valid ranges
- Required model names are specified
- API endpoints are properly formatted
- Resource limits are reasonable

Use the built-in validation:
```python
from athena.core.config import EthicsConfig

config = EthicsConfig("configs/production.yaml")
if config.validate():
    print("Configuration is valid")
else:
    print("Configuration has errors")
```

## Security Considerations

### Sensitive Information
- Never commit API keys or passwords to version control
- Use environment variables or secret management systems
- Rotate keys and credentials regularly

### Access Control
- Limit configuration file access to authorized personnel
- Use proper file permissions (600 for sensitive configs)
- Audit configuration changes

### Compliance
- Production configurations include compliance settings
- Regular audits of threshold adjustments
- Documentation of all configuration changes

## Customization

### Creating Custom Configurations
1. Copy an existing configuration file
2. Modify settings for your specific use case
3. Validate the configuration
4. Test thoroughly before deployment

### Configuration Override Hierarchy
1. Environment variables (highest priority)
2. Command-line arguments
3. Configuration file settings
4. Default values (lowest priority)

## Monitoring and Maintenance

### Configuration Monitoring
- Track configuration changes through version control
- Monitor performance impact of threshold adjustments
- Regular reviews of ethics threshold effectiveness

### Updates and Maintenance
- Quarterly review of all configurations
- Update model references when new versions are available
- Adjust thresholds based on performance data
- Sync staging and production configurations regularly

## Support and Troubleshooting

### Common Issues
- **Model loading failures**: Check model names and paths
- **API connection errors**: Verify endpoints and credentials
- **Performance issues**: Review batch sizes and resource limits
- **Threshold tuning**: Use staging environment for testing

### Getting Help
For configuration issues:
1. Check the logs for specific error messages
2. Validate configuration syntax and values
3. Compare with working configurations
4. Consult the main project documentation

### Contributing
When contributing configuration improvements:
1. Test in development environment first
2. Validate against all environment types
3. Document any new configuration options
4. Update this README with relevant changes