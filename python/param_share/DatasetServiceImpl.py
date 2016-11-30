#!/usr/bin/python
# Copyright (c) 2009 Las Cumbres Observatory (www.lcogt.net)
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

'''HelloWorldServiceImpl.py - 'Hello world' service implementation example.
Authors: Eric Saunders (esaunders@lcogt.net)
         Martin Norbury (mnorbury@lcogt.net)
         Zach Walker (zwalker@lcogt.net)
May 2009
'''

import data_service_pb2
import time


class DataServiceImpl(data_service_pb2.DataShardService):
    def __init__(self,dataset):
        super(data_service_pb2.DataShardService,self).__init__()
        self.dataset=dataset
    def DataService(self, controller, request, done):
        print "In data server"

        # Print the request
        print request

        # Extract name from the message received
        batch_size = int(request.batch_size)
        batchx,batchy = self.dataset.next_batch(batch_size)
        # Create a reply
        response = data_service_pb2.DataShardResponse()
        response.batchx = str(batchx.tobytes())
        response.batchy = str(batchy.tobytes())

        # Sleeping to show asynchronous behavior on client end.
        time.sleep(1)

        # We're done, call the run method of the done callback
        done.run(response)
