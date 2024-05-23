# coding: utf-8

from __future__ import absolute_import

from flask import json
from six import BytesIO

from swagger_server.test import BaseTestCase


class TestBenchmarkController(BaseTestCase):
    """BenchmarkController integration test stubs"""

    def test_create_benchmark(self):
        """Test case for create_benchmark

        Create a benchmark
        """
        response = self.client.open(
            '/api/v3/benchmark',
            method='POST')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_delete_benchmark_byjob_id(self):
        """Test case for delete_benchmark_byjob_id

        Delete Benchmark by jobId
        """
        response = self.client.open(
            '/api/v3/benchmark/{jobId}'.format(job_id=789),
            method='DELETE')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_get_benchmark_byjob_id(self):
        """Test case for get_benchmark_byjob_id

        Get Benchmark by jobId
        """
        response = self.client.open(
            '/api/v3/benchmark/{jobId}'.format(job_id=789),
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_get_benchmarks(self):
        """Test case for get_benchmarks

        Get all Benchmark Jobs
        """
        response = self.client.open(
            '/api/v3/benchmark',
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    import unittest
    unittest.main()
