# coding: utf-8

from __future__ import absolute_import

from flask import json
from six import BytesIO

from swagger_server.test import BaseTestCase


class TestTrainingController(BaseTestCase):
    """TrainingController integration test stubs"""

    def test_create_training(self):
        """Test case for create_training

        Create a training
        """
        response = self.client.open(
            '/api/v3/training',
            method='POST')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_delete_training_byjob_id(self):
        """Test case for delete_training_byjob_id

        Delete Training by jobId
        """
        response = self.client.open(
            '/api/v3/training/{jobId}'.format(job_id=789),
            method='DELETE')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_get_training_byjob_id(self):
        """Test case for get_training_byjob_id

        Get Training by jobId
        """
        response = self.client.open(
            '/api/v3/training/{jobId}'.format(job_id=789),
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_get_trainings(self):
        """Test case for get_trainings

        Get all Training Jobs
        """
        response = self.client.open(
            '/api/v3/training',
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    import unittest
    unittest.main()
