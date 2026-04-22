When generating a g4 changelist description, you must separate the public-facing summary from the private internal details:
1. Place a short, 1-line or brief high-level summary of the change strictly between `BEGIN_PUBLIC` and `END_PUBLIC` tags.
2. Place all detailed technical notes, testing steps, and run instructions *outside* and *below* the `END_PUBLIC` tag so they remain private.

Always use this exact format:

BEGIN_PUBLIC
[1-line or short public summary of the feature or fix]
END_PUBLIC

Details:
- [Private internal technical details]

Testing:
- [Private instructions on how to run and test]
