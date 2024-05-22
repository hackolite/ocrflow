# coding: utf-8

from __future__ import absolute_import

from flask import json
from six import BytesIO

from swagger_server.test import BaseTestCase


class TestMetricController(BaseTestCase):
    """MetricController integration test stubs"""

    def test_get_metrics(self):
        """Test case for get_metrics

        Get All Metrics
        """
        response = self.client.open(
            '/api/v3/metric/',
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_get_metrics_by_id(self):
        """Test case for get_metrics_by_id

        Get Metrics by jobId
        """
        response = self.client.open(
            '/api/v3/metric/{jobId}'.format(job_id=789),
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    import unittest
    unittest.main()
