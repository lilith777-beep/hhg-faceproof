class FaceProofError(Exception):
    """Base class for expected, user-actionable pipeline failures."""


class ConsentRequired(FaceProofError):
    pass


class FaceInputError(FaceProofError):
    pass


class SearchError(FaceProofError):
    pass


class NoVerifiedMatch(SearchError):
    pass


class UnsafeRemoteResource(SearchError):
    pass


class ChainError(FaceProofError):
    pass


class VerificationError(ChainError):
    pass
