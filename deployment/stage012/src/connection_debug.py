import os
import time
import logging
import pandas as pd
import sqlalchemy
from sshtunnel import SSHTunnelForwarder
from contextlib import contextmanager

# Enhanced logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Your existing configuration
ssh_conf = {
    "tunnel_host": "13.201.126.23",
    "tunnel_port": 22,
    "ssh_username": "ec2-user",
}

pg_conf = {
    "host": "10.200.51.243",  
    "port": 3306,
    "dbname": "superset",
    "user": "superset_user",
    "password": "FINadmin123#"
}

def test_ssh_key(key_path):
    """Test if SSH key exists and has correct permissions"""
    print(f"Testing SSH key: {key_path}")
    
    if not os.path.exists(key_path):
        print(f"❌ SSH key file does not exist: {key_path}")
        return False
    
    # Check file permissions
    stat_info = os.stat(key_path)
    permissions = oct(stat_info.st_mode)[-3:]
    print(f"✅ SSH key exists")
    print(f"📁 File permissions: {permissions}")
    
    # SSH keys should typically have 600 or 400 permissions
    if permissions not in ['600', '400']:
        print(f"⚠️  Warning: SSH key permissions are {permissions}, should be 600 or 400")
        print("To fix: chmod 600 /path/to/your/key.pem")
    
    # Check if file is readable
    try:
        with open(key_path, 'r') as f:
            content = f.read(100)  # Read first 100 chars
            if content.startswith('-----BEGIN'):
                print("✅ SSH key appears to be valid")
                return True
            else:
                print("❌ SSH key does not appear to be in correct format")
                return False
    except Exception as e:
        print(f"❌ Cannot read SSH key: {e}")
        return False

def test_ssh_connection(key_path):
    """Test SSH connection without database"""
    print(f"\n🔧 Testing SSH connection to {ssh_conf['tunnel_host']}...")
    
    try:
        tunnel = SSHTunnelForwarder(
            (ssh_conf['tunnel_host'], ssh_conf['tunnel_port']),
            ssh_username=ssh_conf['ssh_username'],
            ssh_pkey=key_path,
            remote_bind_address=(pg_conf['host'], pg_conf['port']),
            set_keepalive=30.0,
            compression=True
        )
        
        print("🚀 Starting SSH tunnel...")
        tunnel.start()
        print(f"✅ SSH tunnel started successfully!")
        print(f"📡 Local bind port: {tunnel.local_bind_port}")
        
        # Test if tunnel is actually working
        time.sleep(2)
        if tunnel.is_alive:
            print("✅ SSH tunnel is alive and healthy")
        else:
            print("❌ SSH tunnel is not responding")
            
        tunnel.stop()
        print("🛑 SSH tunnel stopped")
        return True
        
    except Exception as e:
        print(f"❌ SSH connection failed: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Common error scenarios
        if "Authentication failed" in str(e):
            print("💡 Suggestion: Check SSH key and username")
        elif "Connection refused" in str(e):
            print("💡 Suggestion: Check if SSH host is reachable and SSH service is running")
        elif "No route to host" in str(e):
            print("💡 Suggestion: Check network connectivity and firewall settings")
        elif "Permission denied" in str(e):
            print("💡 Suggestion: Check SSH key permissions and username")
            
        return False

def test_database_connection_direct():
    """Test direct database connection (without SSH tunnel) - for local testing"""
    print(f"\n🗄️  Testing direct database connection...")
    
    try:
        # This will only work if you're on the same network as the DB
        conn_str = f"postgresql://{pg_conf['user']}:{pg_conf['password']}@{pg_conf['host']}:{pg_conf['port']}/{pg_conf['dbname']}"
        engine = sqlalchemy.create_engine(conn_str, connect_args={'connect_timeout': 10})
        
        with engine.connect() as conn:
            result = conn.execute(sqlalchemy.text("SELECT 1 as test"))
            print("✅ Direct database connection successful")
            return True
            
    except Exception as e:
        print(f"❌ Direct database connection failed: {e}")
        print("💡 This is expected if database is behind SSH tunnel")
        return False

@contextmanager
def get_db_connection_enhanced(key_path):
    """Enhanced database connection with detailed error handling"""
    tunnel = None
    engine = None
    
    try:
        print(f"🔧 Creating SSH tunnel to {ssh_conf['tunnel_host']}...")
        
        tunnel = SSHTunnelForwarder(
            (ssh_conf['tunnel_host'], ssh_conf['tunnel_port']),
            ssh_username=ssh_conf['ssh_username'],
            ssh_pkey=key_path,
            remote_bind_address=(pg_conf['host'], pg_conf['port']),
            set_keepalive=30.0,
            compression=True
        )
        
        print("🚀 Starting tunnel...")
        tunnel.start()
        
        # Wait for tunnel to be ready
        max_retries = 5
        for i in range(max_retries):
            if tunnel.is_alive:
                break
            print(f"⏳ Waiting for tunnel... ({i+1}/{max_retries})")
            time.sleep(1)
        
        if not tunnel.is_alive:
            raise ConnectionError("SSH tunnel failed to start properly")
            
        print(f"✅ Tunnel established on local port: {tunnel.local_bind_port}")
        
        # Create database connection
        conn_str = (
            f"postgresql://{pg_conf['user']}:{pg_conf['password']}@"
            f"127.0.0.1:{tunnel.local_bind_port}/{pg_conf['dbname']}"
        )
        
        print("🗄️  Connecting to database...")
        engine = sqlalchemy.create_engine(
            conn_str, 
            pool_timeout=20, 
            pool_recycle=300,
            connect_args={'connect_timeout': 15}
        )
        
        # Test the connection
        with engine.connect() as test_conn:
            test_conn.execute(sqlalchemy.text("SELECT 1"))
            print("✅ Database connection successful!")
        
        yield engine
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        logger.error(f"Connection error: {type(e).__name__}: {e}")
        raise
        
    finally:
        if engine:
            print("🔌 Closing database connection...")
            engine.dispose()
        if tunnel:
            print("🛑 Stopping SSH tunnel...")
            tunnel.stop()

def test_full_workflow(key_path):
    """Test the complete database workflow"""
    print(f"\n🔄 Testing complete database workflow...")
    
    try:
        with get_db_connection_enhanced(key_path) as engine:
            # Test a simple query
            df = pd.read_sql("SELECT 1 as test_col", engine)
            print(f"✅ Query successful, result: {df.iloc[0]['test_col']}")
            
            # Test if your specific table exists
            schema = "epm1-replica.finalyzer.info_100032"  # Your DEFAULT_SCHEMA
            table = "fbi_entity_analysis_report"
            
            try:
                test_query = f'SELECT COUNT(*) as count FROM "{schema}"."{table}" LIMIT 1'
                result = pd.read_sql(test_query, engine)
                print(f"✅ Target table accessible, has {result.iloc[0]['count']} rows")
            except Exception as table_error:
                print(f"⚠️  Target table test failed: {table_error}")
                
        return True
        
    except Exception as e:
        print(f"❌ Full workflow test failed: {e}")
        return False

def diagnose_connection_issues(model_dir):
    """Main diagnostic function"""
    print("🔍 Starting connection diagnostics...\n")
    
    # 1. Check SSH key
    key_path = os.path.join(model_dir, 'data', 'private_key.pem')
    print(f"Expected key path: {key_path}")
    
    if not test_ssh_key(key_path):
        print("❌ SSH key test failed. Please fix SSH key issues first.")
        return False
    
    # 2. Test SSH connection
    if not test_ssh_connection(key_path):
        print("❌ SSH connection test failed.")
        return False
    
    # 3. Test direct DB connection (optional)
    test_database_connection_direct()
    
    # 4. Test full workflow
    if test_full_workflow(key_path):
        print("\n✅ All tests passed! Your connection should work.")
        return True
    else:
        print("\n❌ Some tests failed. Check the errors above.")
        return False

# Alternative simpler connection function for testing
@contextmanager
def get_db_connection_simple(key_path):
    """Simplified connection manager for debugging"""
    tunnel = None
    engine = None
    
    try:
        # Start tunnel with minimal configuration
        tunnel = SSHTunnelForwarder(
            (ssh_conf['tunnel_host'], ssh_conf['tunnel_port']),
            ssh_username=ssh_conf['ssh_username'],
            ssh_pkey=key_path,
            remote_bind_address=(pg_conf['host'], pg_conf['port'])
        )
        
        tunnel.start()
        time.sleep(2)  # Give tunnel time to establish
        
        # Simple connection string
        conn_str = f"postgresql://{pg_conf['user']}:{pg_conf['password']}@127.0.0.1:{tunnel.local_bind_port}/{pg_conf['dbname']}"
        engine = sqlalchemy.create_engine(conn_str)
        
        yield engine
        
    finally:
        if engine:
            engine.dispose()
        if tunnel:
            tunnel.stop()

# Usage example:
if __name__ == "__main__":
    model_dir = "/home/gaurav/finance-to-sql/deployment/stage012/"
    diagnose_connection_issues(model_dir)