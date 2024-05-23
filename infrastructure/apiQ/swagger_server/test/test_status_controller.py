# coding: utf-8

from __future__ import absolute_import

from flask import json
from six import BytesIO

from swagger_server.test import BaseTestCase


class TestStatusController(BaseTestCase):
    """StatusController integration test stubs"""

    def test_get_consumers(self):
        """Test case for get_consumers

        Get All Consumers Status
        """
        response = self.client.open(
            '/api/v3/status/consumers',
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_get_job_byjob_id(self):
        """Test case for get_job_byjob_id

        Get Job by jobId
        """
        response = self.client.open(
            '/api/v3/status/jobs/{jobId}'.format(job_id=789),
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_get_node_bynode_id(self):
        """Test case for get_node_bynode_id

        Get Consumer by nodeId
        """
        response = self.client.open(
            '/api/v3/status/consumers/{nodeId}'.format(node_id=789),
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_gets_jobs(self):
        """Test case for gets_jobs

        Get all Jobs Status
        """
        response = self.client.open(
            '/api/v3/status/jobs',
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    import unittest
    unittest.main()
