#!/usr/bin/python
# Copyright (c) 2009 Las Cumbres Observatory (www.lcogt.net)
# Copyright (c) 2010 Jan Dittberner
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
'''
run_server.py - A simple front-end to demo the RPC server implementation.
Author: Eric Saunders (esaunders@lcogt.net)
        Jan Dittberner (jan@dittberner.info)
May 2009, Nov 2010
'''

# Add main protobuf module to classpath
import sys
sys.path.append('../../protobuf-socket-rpc')

# Import required RPC modules
import protobuf.socketrpc.server as server
import DatasetServiceImpl as impl
from tensorflow.examples.tutorials.mnist import input_data

# Create and register the service
# Note that this is an instantiation of the implementation class,
# *not* the class defined in the proto file.
dataset = input_data.read_data_sets('MNIST_data', one_hot=True).train
data_service = impl.DataserviceImpl(dataset)
server = server.SocketRpcServer(9999)
server.registerService(data_service)

# Start the server
print 'Serving on port 9999'
server.run()
