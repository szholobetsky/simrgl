"""
GitLab connector module for fetching issue details from GitLab instances.
Supports gitlab.com, gitlab.gnome.org, gitlab.freedesktop.org, and self-hosted.
"""

from .gitlab_api_connector import GitLabApiConnector

__all__ = ['GitLabApiConnector']
