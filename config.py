"""
Centralized configuration management for ChatCapitalHumain application.
Handles environment variables and secrets loading.
"""
import os
from typing import Optional
import streamlit as st
from dotenv import load_dotenv

# Load environment variables once at module level
load_dotenv()


class Config:
    """Centralized configuration class for managing environment variables and secrets."""
    
    # Core API Configuration
    OPENROUTER_API_KEY = "OPENROUTER_API_KEY"
    OPENROUTER_BASE_URL = "OPENROUTER_BASE_URL"
    
    # Database Configuration
    DB_URL = "db_url"
    
    # Monitoring and Tracing
    LANGSMITH_API_KEY = "LANGSMITH_API_KEY"
    LANGSMITH_ENDPOINT = "LANGSMITH_ENDPOINT"
    LANGSMITH_PROJECT = "CapitalHumain"
    LANGSMITH_TRACING = True
    
    # Search and Embeddings (actively used)
    VOYAGE_API_KEY = "VOYAGE_API_KEY"
    
    # Authentication Configuration
    AUTH_CLIENT_ID = "AUTH_CLIENT_ID"
    AUTH_CLIENT_SECRET = "AUTH_CLIENT_SECRET"
    AUTH_REDIRECT_URI = "AUTH_REDIRECT_URI"
    AUTH_COOKIE_SECRET = "AUTH_COOKIE_SECRET"
    AUTH_SERVER_METADATA_URL = "AUTH_SERVER_METADATA_URL"
    
    # Debug Settings
    DEBUGGING = False
    
    @staticmethod
    def get_env_variable(var_name: str) -> Optional[str]:
        """
        Get environment variable with fallback to Streamlit secrets.
        
        Args:
            var_name: Name of the environment variable
            
        Returns:
            Value of the environment variable or None if not found
        """
        try:
            # First check environment variables
            if var_name in os.environ:
                return os.environ[var_name]
            
            # Fallback to Streamlit secrets
            if hasattr(st, 'secrets') and var_name in st.secrets:
                return st.secrets[var_name]
                
        except Exception as e:
            st.error(f"Error retrieving environment variable {var_name}: {str(e)}")
        
        return None
    
    @staticmethod
    def ensure_secrets_file():
        """
        Create Streamlit secrets file from environment variables if it doesn't exist.
        This is used for deployment scenarios.
        """
        if not Config.get_env_variable(Config.AUTH_CLIENT_ID):
            return
            
        secrets_path = '.streamlit/secrets.toml'
        if os.path.exists(secrets_path):
            return
            
        os.makedirs('.streamlit', exist_ok=True)
        
        # Required environment variables for secrets file
        required_vars = [
            Config.OPENROUTER_API_KEY,
            Config.OPENROUTER_BASE_URL,
            Config.LANGSMITH_API_KEY,
            Config.LANGSMITH_ENDPOINT,
            Config.VOYAGE_API_KEY,
            Config.DB_URL,
            Config.AUTH_REDIRECT_URI,
            Config.AUTH_COOKIE_SECRET,
            Config.AUTH_CLIENT_ID,
            Config.AUTH_CLIENT_SECRET,
            Config.AUTH_SERVER_METADATA_URL,
        ]
        
        secrets_content = "# API Keys and URLs\n"
        for var in required_vars[:5]:  # API keys section
            value = Config.get_env_variable(var)
            if value:
                secrets_content += f'{var} = "{value}"\n'
        
        secrets_content += f'\nLANGSMITH_TRACING = {str(Config.LANGSMITH_TRACING).lower()}\n'
        secrets_content += f'LANGSMITH_PROJECT = "{Config.LANGSMITH_PROJECT}"\n'
        
        secrets_content += "\n# Database\n"
        db_url = Config.get_env_variable(Config.DB_URL)
        if db_url:
            secrets_content += f'{Config.DB_URL} = "{db_url}"\n'
        
        secrets_content += f"\n# Debug settings\nDEBUGGING = {str(Config.DEBUGGING).lower()}\n"
        
        secrets_content += "\n[auth]\n"
        auth_vars = required_vars[6:]  # Auth variables
        for var in auth_vars:
            value = Config.get_env_variable(var)
            if value:
                # Remove AUTH_ prefix for secrets file
                key = var.lower().replace('auth_', '')
                secrets_content += f'{key} = "{value}"\n'
        
        try:
            with open(secrets_path, 'w') as f:
                f.write(secrets_content)
        except Exception as e:
            st.error(f"Failed to create secrets file: {str(e)}")
    
    @staticmethod
    def get_openrouter_config() -> dict:
        """Get OpenRouter API configuration."""
        return {
            "openai_api_key": Config.get_env_variable(Config.OPENROUTER_API_KEY),
            "openai_api_base": Config.get_env_variable(Config.OPENROUTER_BASE_URL),
        }
    
    @staticmethod
    def get_database_url() -> Optional[str]:
        """Get database URL."""
        return Config.get_env_variable(Config.DB_URL)
    
    @staticmethod
    def get_voyage_api_key() -> Optional[str]:
        """Get Voyage AI API key."""
        return Config.get_env_variable(Config.VOYAGE_API_KEY)
    
    @staticmethod
    def is_debugging() -> bool:
        """Check if debugging mode is enabled."""
        debug_value = Config.get_env_variable("DEBUGGING")
        if debug_value:
            return debug_value.lower() in ['true', '1', 'yes', 'on']
        return Config.DEBUGGING


# Initialize configuration on import
Config.ensure_secrets_file()
