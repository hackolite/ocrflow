# coding: utf-8

from __future__ import absolute_import

from flask import json
from six import BytesIO

from swagger_server.test import BaseTestCase


class TestManageController(BaseTestCase):
    """ManageController integration test stubs"""

    def test_get_datasets(self):
        """Test case for get_datasets

        Manage Datasets
        """
        response = self.client.open(
            '/api/v3/manage/datasets',
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_get_exchanges(self):
        """Test case for get_exchanges

        Manage Exchanges
        """
        response = self.client.open(
            '/api/v3/manage/exchange',
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_get_models(self):
        """Test case for get_models

        Manage Models
        """
        response = self.client.open(
            '/api/v3/manage/models',
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_get_queues(self):
        """Test case for get_queues

        Manage Queues
        """
        response = self.client.open(
            '/api/v3/manage/queues',
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    import unittest
    unittest.main()
