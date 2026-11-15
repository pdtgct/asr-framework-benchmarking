"""Builders that turn a locally staged corpus into an asrbench manifest.

A builder reads files already on disk. It never fetches, so the same
command works on an air-gapped host once the corpus is staged. The first
supported corpus is AMI (:mod:`asrbench.corpora.ami`); anything that can
produce :data:`asrbench.manifest.MANIFEST_SCHEMA` JSON plugs in the same
way, because the runner only ever reads the manifest.
"""
