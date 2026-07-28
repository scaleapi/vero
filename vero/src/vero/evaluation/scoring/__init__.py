"""Turning raw results into a trustworthy number.

``objective`` resolves a metric and compares candidates, ``error_taxonomy``
classifies failures so infrastructure noise is not scored as a bad candidate, and
``security`` redacts secrets from anything shown to the agent.
"""
