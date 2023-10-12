def is_palindrome(s):
    """ Assumes s is a str
    returns True if letters in s form a palindrome; False
    otherwise. Non-letters and capitalization are ignored."""
    
    def to_chars(s):
        s = s.lower()
        letters = ""
        
        for c in s:
            if c in "abcdefghijklmnopqrstuvwxyz":
                letters += c
        
        return letters
    
    def is_pal(s):
        if len(s) <= 1:
            return True
        
        else:
            return s[0] == s[-1] and is_pal(s[1:-1])
        
    return is_pal(to_chars(s))