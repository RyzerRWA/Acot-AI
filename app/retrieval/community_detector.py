class CommunityDetector:

    def __init__(self):

        self.communities = [
            "Dubai Marina",
            "Downtown Dubai",
            "Business Bay",
            "Palm Jumeirah",
            "Dubai Hills Estate"
        ]


    def detect(self, question: str):

        question_lower = question.lower()

        for community in self.communities:

            if community.lower() in question_lower:

                return community

        return None