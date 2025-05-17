from haystack import component

@component
class UserInfo:
    def __init__(self, userdict = dict()):
        self.userdict =  userdict
    
    @component.output_types(user_info=dict)
    def run(self, id: int = 0):
        return {"user_info": self.userdict}
