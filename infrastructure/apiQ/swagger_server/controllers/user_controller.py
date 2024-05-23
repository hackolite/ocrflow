import connexion
import six

from swagger_server.models.user import User  # noqa: E501
from swagger_server import util


def create_user(body=None):  # noqa: E501
    """Create user

    This can only be done by the logged in user. # noqa: E501

    :param body: Created user object
    :type body: dict | bytes

    :rtype: User
    """
    if connexion.request.is_json:
        body = User.from_dict(connexion.request.get_json())  # noqa: E501
    return 'do some magic!'


def create_user(id=None, username=None, first_name=None, last_name=None, email=None, password=None, phone=None, user_status=None):  # noqa: E501
    """Create user

    This can only be done by the logged in user. # noqa: E501

    :param id: 
    :type id: int
    :param username: 
    :type username: str
    :param first_name: 
    :type first_name: str
    :param last_name: 
    :type last_name: str
    :param email: 
    :type email: str
    :param password: 
    :type password: str
    :param phone: 
    :type phone: str
    :param user_status: 
    :type user_status: int

    :rtype: User
    """
    return 'do some magic!'


def create_users_with_list_input(body=None):  # noqa: E501
    """Creates list of users with given input array

    Creates list of users with given input array # noqa: E501

    :param body: 
    :type body: list | bytes

    :rtype: User
    """
    if connexion.request.is_json:
        body = [User.from_dict(d) for d in connexion.request.get_json()]  # noqa: E501
    return 'do some magic!'


def delete_user(username):  # noqa: E501
    """Delete user

    This can only be done by the logged in user. # noqa: E501

    :param username: The name that needs to be deleted
    :type username: str

    :rtype: None
    """
    return 'do some magic!'


def get_user_by_name(username):  # noqa: E501
    """Get user by user name

     # noqa: E501

    :param username: The name that needs to be fetched. Use user1 for testing. 
    :type username: str

    :rtype: User
    """
    return 'do some magic!'


def login_user(username=None, password=None):  # noqa: E501
    """Logs user into the system

     # noqa: E501

    :param username: The user name for login
    :type username: str
    :param password: The password for login in clear text
    :type password: str

    :rtype: str
    """
    return 'do some magic!'


def logout_user():  # noqa: E501
    """Logs out current logged in user session

     # noqa: E501


    :rtype: None
    """
    return 'do some magic!'


def update_user(username, body=None):  # noqa: E501
    """Update user

    This can only be done by the logged in user. # noqa: E501

    :param username: name that need to be deleted
    :type username: str
    :param body: Update an existent user in the store
    :type body: dict | bytes

    :rtype: None
    """
    if connexion.request.is_json:
        body = User.from_dict(connexion.request.get_json())  # noqa: E501
    return 'do some magic!'


def update_user(username, id=None, username2=None, first_name=None, last_name=None, email=None, password=None, phone=None, user_status=None):  # noqa: E501
    """Update user

    This can only be done by the logged in user. # noqa: E501

    :param username: name that need to be deleted
    :type username: str
    :param id: 
    :type id: int
    :param username2: 
    :type username2: str
    :param first_name: 
    :type first_name: str
    :param last_name: 
    :type last_name: str
    :param email: 
    :type email: str
    :param password: 
    :type password: str
    :param phone: 
    :type phone: str
    :param user_status: 
    :type user_status: int

    :rtype: None
    """
    return 'do some magic!'
