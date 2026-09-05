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


class BlockedState(FaceProofError):
    """A named external prerequisite is unavailable; never a passed or skipped check."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")
