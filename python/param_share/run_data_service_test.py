'''run_client.py - A simple front-end to demo the RPC client implementation.
Authors: Eric Saunders (esaunders@lcogt.net)
         Martin Norbury (mnorbury@lcogt.net)
         Zach Walker (zwalker@lcogt.net)
         Jan Dittberner (jan@dittberner.info)
May 2009, Nov 2010
'''

# Add main protobuf module to classpath
import sys
sys.path.append('../../protobuf-socket-rpc/python/src/')

# Import required RPC modules
import data_service_pb2 
from protobuf.socketrpc import RpcService

# Configure logging
import logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# Server details
hostname = 'localhost'
port = 9999


# Create a request
request = data_service_pb2.DataShardRequest()
request.batch_size = '10'

# Create a new service instance
service = RpcService(data_service_pb2.DataShardService_Stub,
                     port,
                     hostname)

# Make a synchronous call
try:
    log.info('Making synchronous call')
    response = service.DataService(request, timeout=1000)
    log.info('Synchronous response: ' + response.__str__())
except Exception, ex:
    log.exception(ex)
