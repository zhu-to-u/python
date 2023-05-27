def play_alice(n):
    if n==0:
        print("Bob wins!")
    else:
        play_bob(n-1)

def play_bob(n):
    if n==0:
        print("Alice wins!")
    elif is_even(n):
        play_alice(n-2)
    else:
        play_alice(n-1)

def is_even(n):
    if n%2==0:
        return 1
    else:
        return 0
    