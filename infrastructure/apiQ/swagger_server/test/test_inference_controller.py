# coding: utf-8

from __future__ import absolute_import

from flask import json
from six import BytesIO

from swagger_server.test import BaseTestCase


class TestInferenceController(BaseTestCase):
    """InferenceController integration test stubs"""

    def test_create_inference(self):
        """Test case for create_inference

        Create Inference
        """
        response = self.client.open(
            '/api/v3/inference',
            method='POST')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_delete_inference_by_job_id(self):
        """Test case for delete_inference_by_job_id

        Delete Inference By jobId
        """
        response = self.client.open(
            '/api/v3/inference/{jobId}'.format(job_id=789),
            method='DELETE')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_find_inferences_by_status(self):
        """Test case for find_inferences_by_status

        Find Inferences by status
        """
        query_string = [('status', 'waiting')]
        response = self.client.open(
            '/api/v3/inference/findByStatus',
            method='GET',
            query_string=query_string)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_find_inferences_by_tags(self):
        """Test case for find_inferences_by_tags

        Find Inferences by tags
        """
        query_string = [('tags', 'tags_example')]
        response = self.client.open(
            '/api/v3/inference/findByTags',
            method='GET',
            query_string=query_string)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_get_inference_byjob_id(self):
        """Test case for get_inference_byjob_id

        Find Inference job by jobId
        """
        response = self.client.open(
            '/api/v3/inference/{jobId}'.format(job_id=789),
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_get_inferences(self):
        """Test case for get_inferences

        Get all Inference Jobs
        """
        response = self.client.open(
            '/api/v3/inference',
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_update_inference_by_job_id(self):
        """Test case for update_inference_by_job_id

        Updates Inference by jobId
        """
        query_string = [('name', 'name_example'),
                        ('status', 'status_example')]
        response = self.client.open(
            '/api/v3/inference/{jobId}'.format(job_id=789),
            method='PUT',
            query_string=query_string)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    import unittest
    unittest.main()
