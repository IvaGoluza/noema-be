# Deployment-realistic examples.
# Each example is a single thought as would appear in the "automatic thoughts"
# field of the application after being prompted by a situation. They are NOT
# taken from any test set.
#
# Source: tqa_pr_pipeline_arch_deploy_test.ipynb, cell 1.
# Also documented in Appendix A of the thesis.

EXAMPLES = [
    # ===== Obviously distorted (one per category) =====
    # All-or-nothing
    ("distorted", "If I don't get this promotion, my whole career is a failure.",
     "All-or-nothing thinking"),
    # Catastrophizing
    ("distorted", "I made one mistake in the presentation and now everyone will think I'm incompetent.",
     "Catastrophizing"),
    # Discounting the positive
    ("distorted", "She said she liked my work but she was probably just being polite.",
     "Discounting the positive"),
    # Fortune-telling
    ("distorted", "I just know this date is going to go badly.",
     "Fortune-telling"),
    # Mind reading
    ("distorted", "He didn't text me back, so he must be angry with me.",
     "Mind reading"),
    # Labeling
    ("distorted", "I forgot to send the email. I'm such an idiot.",
     "Labeling"),
    # Mental filter
    ("distorted", "Everyone clapped except for one person who looked bored. The whole talk was a disaster.",
     "Mental filtering"),
    # Overgeneralization
    ("distorted", "I bombed this job interview. I'm never going to find work.",
     "Overgeneralization"),
    # Personalization
    ("distorted", "My partner is in a bad mood. It must be something I did.",
     "Personalization"),
    # Should statements
    ("distorted", "I should be further along in my career by now.",
     "Should statements"),

    # ===== non-distorted =====
    ("nondistorted", "I should call my mother this weekend.", "No Distortion"),
    ("nondistorted", "It looks like it might rain later.", "No Distortion"),
    ("nondistorted", "I have a meeting at 3pm tomorrow.", "No Distortion"),
    ("nondistorted", "The coffee tastes good today.", "No Distortion"),
    ("nondistorted", "I want to read more this year.", "No Distortion"),
    ("nondistorted", "My back hurts after sitting all day.", "No Distortion"),
    ("nondistorted", "I'm feeling tired and I think I'll go to bed early.", "No Distortion"),
    ("nondistorted", "The deadline is on Friday so I need to finish the report by Thursday.", "No Distortion"),
    ("nondistorted", "I disagreed with what she said but I understand her point.", "No Distortion"),
    ("nondistorted", "I made some progress today, though there's still more to do.", "No Distortion"),

    # ===== Ambiguous / borderline =====
    # Could be a fact or a distortion depending on intent
    ("ambiguous", "He didn't say hi to me at the meeting.", "uncertain"),
    # Genuine concern vs catastrophizing — depends on context
    ("ambiguous", "I'm worried I won't have enough money for retirement.", "uncertain"),
    # Realistic expectation vs all-or-nothing
    ("ambiguous", "If the funding doesn't come through, the project is over.", "uncertain"),
    # Reasonable inference vs mind reading
    ("ambiguous", "My boss has been avoiding eye contact this week.", "uncertain"),
    # Self-criticism that may be deserved
    ("ambiguous", "I shouldn't have lost my temper with him.", "uncertain"),
    # Reasonable concern vs fortune-telling
    ("ambiguous", "Given the market right now, jobs in my field are going to be hard to find.", "uncertain"),
    # Mixed positive and negative
    ("ambiguous", "I did well overall but I keep thinking about that one bad answer.", "uncertain"),
    # Reasonable doubt
    ("ambiguous", "I don't think I'm qualified for this role.", "uncertain"),
    # Self-blame that may or may not be appropriate
    ("ambiguous", "The team missed the deadline and I was the lead.", "uncertain"),
    # General malaise
    ("ambiguous", "Something doesn't feel right but I can't put my finger on what.", "uncertain"),
]


if __name__ == "__main__":
    print(f"Total examples: {len(EXAMPLES)}")
    print(f"  Obviously distorted:     {sum(1 for x in EXAMPLES if x[0] == 'distorted')}")
    print(f"  Obviously non-distorted: {sum(1 for x in EXAMPLES if x[0] == 'nondistorted')}")
    print(f"  Ambiguous / borderline:  {sum(1 for x in EXAMPLES if x[0] == 'ambiguous')}")
