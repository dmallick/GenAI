import pytest
# Assuming your langchain.py has some functions to test
# For example, if langchain.py has a function called process_text
# from your_package import langchain # If langchain.py is part of a package
import Test # if langchain.py is in the same directory

def test_hello_world():
    assert Test.hello() == "hello"
