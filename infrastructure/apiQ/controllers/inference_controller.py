import connexion
import six

from swagger_server import util


def create_inference():  # noqa: E501
    """Create Inference

    Create Inference Job # noqa: E501


    :rtype: None
    """
    return 'do some magic!'


def delete_inference_by_job_id(job_id):  # noqa: E501
    """Delete Inference By jobId

    delete Inference By jobId # noqa: E501

    :param job_id: Job id to delete
    :type job_id: int

    :rtype: None
    """
    return 'do some magic!'


def find_inferences_by_status(status=None):  # noqa: E501
    """Find Inferences by status

    Multiple status values can be provided with comma separated strings # noqa: E501

    :param status: Status values that need to be considered for filter
    :type status: str

    :rtype: None
    """
    return 'do some magic!'


def find_inferences_by_tags(tags=None):  # noqa: E501
    """Find Inferences by tags

    Multiple tags can be provided with comma separated strings. Use tag1, tag2, tag3 for testing. # noqa: E501

    :param tags: Tags to filter by
    :type tags: List[str]

    :rtype: None
    """
    return 'do some magic!'


def get_inference_byjob_id(job_id):  # noqa: E501
    """Find Inference job by jobId

    get inference by jobId # noqa: E501

    :param job_id: ID of job to return
    :type job_id: int

    :rtype: None
    """
    return 'do some magic!'


def get_inferences():  # noqa: E501
    """Get all Inference Jobs

    Retrieve all inferences # noqa: E501


    :rtype: None
    """
    return 'do some magic!'


def update_inference_by_job_id(job_id, name=None, status=None):  # noqa: E501
    """Updates Inference by jobId

    update inference by jobId # noqa: E501

    :param job_id: ID of pet that needs to be updated
    :type job_id: int
    :param name: Name of jobId that needs to be updated
    :type name: str
    :param status: Status of jobId that needs to be updated
    :type status: str

    :rtype: None
    """
    return 'do some magic!'
