# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
From https://github.com/sail-sg/understand-r1-zero/blob/main/understand_r1_zero/math_grader.py

Provides a math answer grading function with high recall.
Based on HF math_verify, verl, open reasoner zero, etc.
"""

import re
import signal
from itertools import islice, zip_longest
from math import isclose
from typing import Optional

import sympy
from latex2sympy2_extended import latex2sympy
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
from pylatexenc import latex2text
from sympy import N, simplify
from sympy.parsing import sympy_parser
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr


# Dan Hendrycks' code
def mathd_normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        # Remove enclosing `\text{}`.
        m = re.search("^\\\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except:
        return answer


# units mainly from MathQA
unit_texts = [
    "east",
    "degree",
    "mph",
    "kmph",
    "ft",
    "m sqaure",
    " m east",
    "sq m",
    "deg",
    "mile",
    "q .",
    "monkey",
    "prime",
    "ratio",
    "profit of rs",
    "rd",
    "o",
    "gm",
    "p . m",
    "lb",
    "tile",
    "per",
    "dm",
    "lt",
    "gain",
    "ab",
    "way",
    "west",
    "a .",
    "b .",
    "c .",
    "d .",
    "e .",
    "f .",
    "g .",
    "h .",
    "t",
    "a",
    "h",
    "no change",
    "men",
    "soldier",
    "pie",
    "bc",
    "excess",
    "st",
    "inches",
    "noon",
    "percent",
    "by",
    "gal",
    "kmh",
    "c",
    "acre",
    "rise",
    "a . m",
    "th",
    "π r 2",
    "sq",
    "mark",
    "l",
    "toy",
    "coin",
    "sq . m",
    "gallon",
    "° f",
    "profit",
    "minw",
    "yr",
    "women",
    "feet",
    "am",
    "pm",
    "hr",
    "cu cm",
    "square",
    "v â € ™",
    "are",
    "rupee",
    "rounds",
    "cubic",
    "cc",
    "mtr",
    "s",
    "ohm",
    "number",
    "kmph",
    "day",
    "hour",
    "minute",
    "min",
    "second",
    "man",
    "woman",
    "sec",
    "cube",
    "mt",
    "sq inch",
    "mp",
    "∏ cm ³",
    "hectare",
    "more",
    "sec",
    "unit",
    "cu . m",
    "cm 2",
    "rs .",
    "rs",
    "kg",
    "g",
    "month",
    "km",
    "m",
    "cm",
    "mm",
    "apple",
    "liter",
    "loss",
    "yard",
    "pure",
    "year",
    "increase",
    "decrease",
    "d",
    "less",
    "Surface",
    "litre",
    "pi sq m",
    "s .",
    "metre",
    "meter",
    "inch",
]

unit_texts.extend([t + "s" for t in unit_texts])


def _strip_string(string):
    def _fix_fracs(string):
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
        string = new_str
        return string

    def _fix_a_slash_b(string):
        if len(string.split("/")) != 2:
            return string
        a = string.split("/")[0]
        b = string.split("/")[1]
        try:
            a = int(a)
            b = int(b)
            assert string == f"{a}/{b}"
            new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
            return new_string
        except:
            return string

    def _remove_right_units(string):
        # "\\text{ " only ever occurs (at least in the val set) when describing units
        if "\\text{ " in string:
            splits = string.split("\\text{ ")
            assert len(splits) == 2
            return splits[0]
        else:
            return string

    def _fix_sqrt(string):
        if "\\sqrt" not in string:
            return string
        splits = string.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
        return new_string

    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # matrix
    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = (
        string.replace("\\neq", "\\ne")
        .replace("\\leq", "\\le")
        .replace("\\geq", "\\ge")
    )
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    # Remove unit: texts
    for _ in range(2):
        for unit_text in unit_texts:
            # use regex, the prefix should be either the start of the string or a non-alphanumeric character
            # the suffix should be either the end of the string or a non-alphanumeric character
            _string = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", string)
            if _string != "":
                string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]


REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.
    This code comes from https://arxiv.org/pdf/2206.14858.pdf, page18.
    """
    # final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    # \fracab -> \frac{a}{b}
    # \frac{abc}{bef} -> \frac{abc}{bef}
    # \fracabc -> \frac{a}{b}c
    # \sqrta -> \sqrt{a}
    # \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer


def repeatness(s: str):
    def ranks(l):
        index = {v: i for i, v in enumerate(sorted(set(l)))}
        return [index[v] for v in l]

    def suffixArray(s):
        line = ranks(s)
        n, k, ans, sa = len(s), 1, line, [0] * len(s)
        while k < n - 1:
            line = ranks(list(zip_longest(line, islice(line, k, None), fillvalue=-1)))
            ans, k = line, k << 1
        for i, k in enumerate(ans):
            sa[k] = i
        return ans, sa

    def lcp(arr, suffixArr, inv_suff):
        n, ans, k = len(arr), [0] * len(arr), 0

        for i in range(n):
            if inv_suff[i] == n - 1:
                k = 0
                continue

            j = suffixArr[inv_suff[i] + 1]
            while i + k < n and j + k < n and arr[i + k] == arr[j + k]:
                k += 1

            ans[inv_suff[i]] = k
            if k > 0:
                k -= 1

        return ans

    arr = [ord(i) for i in s]
    n = len(arr)
    if n <= 1:
        return 0
    c, sa = suffixArray(arr)
    cnt = sum(lcp(arr, sa, c))

    return (cnt * 2 / (n * (n + 1))) > 0.2


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def latex_eval(latex):
    sym = parse_latex(latex)
    val = sym.evalf()
    return sym, val


def numeric_equal(prediction: float, reference: float):
    # Note that relative tolerance has significant impact
    # on the result of the synthesized GSM-Hard dataset
    # if reference.is_integer():
    #     return isclose(reference, round(prediction), abs_tol=1e-4)
    # else:
    # prediction = round(prediction, len(str(reference).split(".")[-1]))
    return isclose(reference, prediction, rel_tol=1e-4)


def symbolic_equal(a, b):
    def _parse(s):
        for f in [parse_latex, parse_expr, latex2sympy]:
            try:
                return f(s.replace("\\\\", "\\"))
            except:
                try:
                    return f(s)
                except:
                    pass
        return s

    a = _parse(a)
    b = _parse(b)

    # direct equal
    try:
        if str(a) == str(b) or a == b:
            return True
    except:
        pass

    # simplify equal
    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except:
        pass

    # equation equal
    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except:
        pass

    try:
        if numeric_equal(float(N(a)), float(N(b))):
            return True
    except:
        pass

    # matrix
    try:
        # if a and b are matrix
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except:
        pass

    return False


def _is_latex_equal(str1, str2):
    try:
        sym1, val1 = latex_eval(str1)
        sym2, val2 = latex_eval(str2)
        if sym1 == sym2 or val1 == val2:
            return True
        else:
            raise ValueError
    except Exception:  # noqa
        try:
            norm1, norm2 = normalize_final_answer(str1), normalize_final_answer(str2)
            sym1, val1 = latex_eval(norm1)
            sym2, val2 = latex_eval(norm2)
            if sym1 == sym2 or val1 == val2:
                return True
        except Exception:  # noqa
            return norm1 == norm2
    return False


def is_latex_equal(given_answer: str, ground_truth: str) -> bool:
    try:
        with timeout(1):
            try:
                if (len(given_answer) > 128 and repeatness(given_answer)) or (
                    len(ground_truth) > 128 and repeatness(ground_truth)
                ):
                    return False
                # First conduct normalized string matching.
                ground_truth_normalized = _normalize(ground_truth)
                given_normalized = _normalize(given_answer)
                if ground_truth_normalized is None:
                    return False
                if ground_truth_normalized == given_normalized:
                    return True

                # Next call math verify.
                given_answer.replace("\n", "")
                ground_truth.replace("\n", "")
                if "$" not in given_answer:
                    given_answer = f"${given_answer}$"
                if "$" not in ground_truth:
                    ground_truth = f"${ground_truth}$"
                return verify(
                    parse(
                        ground_truth,
                        extraction_config=(
                            LatexExtractionConfig(boxed_match_priority=0),
                            ExprExtractionConfig(),
                        ),
                        fallback_mode="no_fallback",
                        extraction_mode=["first_match"],
                        parsing_timeout=1,
                    ),
                    parse(
                        given_answer,
                        extraction_config=(
                            LatexExtractionConfig(boxed_match_priority=0),
                            ExprExtractionConfig(),
                        ),
                        fallback_mode="no_fallback",
                        extraction_mode=["first_match"],
                        parsing_timeout=1,
                    ),
                    timeout_seconds=1,
                )
                # or symbolic_equal(ground_truth, given_answer)
            except Exception:
                return False
    except TimeoutError:
        return False


def is_value_equal(given_answer: str, ground_truth: str) -> bool:
    assert ground_truth is not None
    ground_truth_normalized_mathd = mathd_normalize_answer(ground_truth)
    given_answer_normalized_mathd = mathd_normalize_answer(given_answer)

    str_equal = ground_truth_normalized_mathd == given_answer_normalized_mathd
    try:
        number_equal = float(ground_truth_normalized_mathd) == float(
            given_answer_normalized_mathd
        )
        return str_equal or number_equal
    except Exception:
        return str_equal


# sympy might hang -- we don't care about trying to be lenient in these cases
BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = ["\^[0-9]+\^", "\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    """Parses an expression with sympy."""
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _parse_latex(expr: str) -> str:
    """Attempts to parse latex to an expression sympy can read."""
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    # Replace the specific characters that this parser uses.
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")

    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _str_to_int(x: str) -> bool:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)  ## implicit mults
    return step


def _strip_properly_formatted_commas(expr: str):
    # We want to be careful because we don't want to strip tuple commas
    p1 = re.compile("(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize(expr: str) -> str:
    """Normalize answer expressions."""
    if expr is None:
        return None

    # Remove enclosing `\text{}`.
    m = re.search("^\\\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
    ]:
        expr = re.sub(f"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub("\^ *\\\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except:
            pass

    # edge case with mixed numbers and negative signs
    expr = re.sub("- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")

    # if we somehow still have latex braces here, just drop them
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")

    # don't be case sensitive for text answers
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def should_allow_eval(expr: str):
    # we don't want to try parsing unknown text or functions of more than two variables
    if count_unknown_letters_in_expr(expr) > 2:
        return False

    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False

    return True


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except:
        pass
    return are_equal


def split_tuple(expr: str):
    """
    Split the elements in a tuple/interval, while handling well-formatted commas in large numbers
    """
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except:
        return None


def extract_boxed_answer(solution: str) -> str:
    """Extract the answer from inside a LaTeX \\boxed{} command"""
    solution = last_boxed_only_string(solution)
    solution = remove_boxed(solution)
    return solution


def grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    if ground_truth_normalized is None:
        return False

    if ground_truth_normalized == given_normalized:
        return True

    if len(given_normalized) == 0:
        return False

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if len(ground_truth_elems) > 1 and (
        ground_truth_normalized[0] != given_normalized[0]
        or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                # if fractions aren't reduced, then shouldn't be marked as correct
                # so, we don't want to allow sympy.simplify in this case
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                # if the ground truth answer is an integer, we require the given answer to be a strict match (no sympy.simplify)
                is_correct = False
            else:
                is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
            if not is_correct:
                break

    return is_correct


def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized_mathd = mathd_normalize_answer(ground_truth)
    given_answer_normalized_mathd = mathd_normalize_answer(given_answer)

    # be at least as lenient as mathd
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True
    return False


def extract_answer(passage: str) -> str:
    if "\\boxed" in passage:
        return extract_boxed_answer(passage)
    return None


def grade(model_answer: str, gt_answer: str, fast: bool = True):
    if "\\boxed" in gt_answer:
        gt_answer = extract_answer(gt_answer)
    correct = grade_answer_mathd(model_answer, gt_answer) or grade_answer_sympy(
        model_answer, gt_answer
    )
    if not fast:
        # This mode further uses math_verify to recall originally false positives.
        # Will be a bit slower, and sensitive to bad inputs.
        correct = correct or is_latex_equal(
            model_answer,
            gt_answer,
        )
    return correct


def r1_zero_reward_fn(response, ground_truth, fast=True):
    # We are strict about format to evaluate our models.
    if "</think> <answer>" in response and "</answer>" in response:
        model_answer = response.split("<answer>")[-1].replace("</answer>", "")
        if "\\boxed" in model_answer:
            model_answer = extract_answer(model_answer)
            if model_answer is None:
                return {
                    "format_reward": 1.0,
                    "answer_reward": 0.0,
                    "reward": 0.0
                }
        if isinstance(ground_truth, float) or isinstance(ground_truth, int):
            ground_truth = str(ground_truth)
        if isinstance(ground_truth, str):
            is_correct = grade(model_answer, ground_truth, fast)
        elif isinstance(ground_truth, list):
            is_correct = False
            for gt in ground_truth:
                is_correct |= grade(model_answer, gt, fast)
        if is_correct:
            return {
                "format_reward": 1.0,
                "answer_reward": 1.0,
                "reward": 1.0
            }
        else:
            # Formatted but wrong answer; no format reward to avoid hacking.
            return {
                "format_reward": 1.0,
                "answer_reward": 0.0,
                "reward": 0.0
            }
    else:
        # Unformatted.
        return {
            "format_reward": 0.0,
            "answer_reward": 0.0,
            "reward": 0.0
        }


def question_only_reward_fn(response, ground_truth, fast=True):
    model_answer = extract_answer(response)
    if model_answer is None:
        # Cannot even parse anything.
        return {
            "format_reward": 0.0,
            "answer_reward": 0.0,
            "reward": 0.0
        }
    if isinstance(ground_truth, float) or isinstance(ground_truth, int):
        ground_truth = str(ground_truth)
    if isinstance(ground_truth, str):
        is_correct = grade(model_answer, ground_truth, fast)
    elif isinstance(ground_truth, list):
        is_correct = False
        for gt in ground_truth:
            is_correct |= grade(model_answer, gt, fast)
    if is_correct:
        # Correctness reward.
        return {
            "format_reward": 1.0,
            "answer_reward": 1.0,
            "reward": 1.0
        }
    else:
        # Formatted but wrong answer; no format reward to avoid hacking.
        return {
            "format_reward": 1.0,
            "answer_reward": 0.0,
            "reward": 0.0
        }



import json
import os
import re
from typing import List, Callable, Dict, Any
from pathlib import Path

from vllm import LLM, SamplingParams


def load_math_validation_data(file_path: str) -> List[Dict[str, Any]]:
    """
    加载 MATH 验证数据集
    
    Args:
        file_path: MATH 验证数据集的路径
        
    Returns:
        包含所有数学问题的列表
    """
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    return examples


def extract_final_answer(answer_text: str) -> str:
    """
    从答案文本中提取最终答案（#### 后面的部分）
    
    Args:
        answer_text: 包含推理过程和最终答案的文本
        
    Returns:
        提取的最终答案
    """
    # 使用正则表达式匹配 #### 后面的内容
    match = re.search(r'####\s*([^\n]+)', answer_text)
    if match:
        return match.group(1).strip()
    else:
        # 如果没有找到 #### 标记，返回原始文本
        return answer_text.strip()


def format_r1_zero_prompt(problem: str) -> str:
    """
    使用 r1_zero 提示格式格式化数学问题
    
    Args:
        problem: 原始数学问题
        
    Returns:
        格式化后的提示
    """
    # 从文件读取提示模板
    prompt_file_path = "./cs336_alignment/prompts/r1_zero.prompt"
    
    try:
        with open(prompt_file_path, 'r', encoding='utf-8') as file:
            prompt_template = file.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"提示文件未找到: {prompt_file_path}")
    except Exception as e:
        raise Exception(f"读取提示文件时出错: {e}")
    
    # 格式化提示
    formatted_prompt = prompt_template.format(question=problem)
    return formatted_prompt


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], Dict[str, float]],
    prompts: List[str],
    gold_answers: List[str],
    eval_sampling_params: SamplingParams,
    output_file: str
) -> Dict[str, Any]:
    """
    评估语言模型在提示列表上的表现，计算评估指标并将结果序列化到磁盘
    
    Args:
        vllm_model: 初始化的 vLLM 模型
        reward_fn: 奖励函数，接受模型输出和真实答案，返回评分字典
        prompts: 提示列表
        gold_answers: 对应的真实答案列表
        eval_sampling_params: 采样参数
        output_file: 输出文件路径
        
    Returns:
        包含评估指标的字典
    """
    print(f"Generating responses for {len(prompts)} prompts...")
    
    # 使用 vLLM 生成响应
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    
    results = []
    correct_count = 0
    format_correct_count = 0
    
    print("Evaluating responses...")
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        gold_answer = gold_answers[i]
        
        # 计算奖励/正确性 - 根据实际的 r1_zero_reward_fn 调整
        reward_result = reward_fn(generated_text, gold_answer)
        
        # 根据实际的奖励函数结构判断正确性
        is_correct = reward_result.get('reward', 0.0) == 1.0
        is_format_correct = reward_result.get('format_reward', 0.0) == 1.0
        
        if is_correct:
            correct_count += 1
        if is_format_correct:
            format_correct_count += 1
        
        # 保存结果
        result = {
            'prompt': prompt,
            'generated_text': generated_text,
            'gold_answer': gold_answer,
            'is_correct': is_correct,
            'is_format_correct': is_format_correct,
            'reward_result': reward_result,
            'example_index': i
        }
        results.append(result)

        # ✅ 每隔 40 个打印一次示例
        if (i + 1) % 40 == 0 or i == 0:
            print("\n" + "=" * 80)
            print(f"[Example {i+1}/{len(outputs)}]")
            print("- Prompt:")
            print(prompt[:1000])
            print("- Model Output:")
            print(generated_text[:1000])
            print(f"- Gold Answer: {gold_answer}")
            print(f"- Reward Result: {reward_result}")
            print(f"- Correct: {is_correct}, Format OK: {is_format_correct}")
            print("=" * 80 + "\n")
    
    # 计算评估指标
    total_examples = len(results)
    accuracy = correct_count / total_examples if total_examples > 0 else 0.0
    format_accuracy = format_correct_count / total_examples if total_examples > 0 else 0.0
    
    metrics = {
        'total_examples': total_examples,
        'correct_count': correct_count,
        'format_correct_count': format_correct_count,
        'accuracy': accuracy,
        'format_accuracy': format_accuracy,
        'model_name': 'Qwen2.5-Math-1.5B',
        'evaluation_type': 'zero_shot_math'
    }
    
    # 序列化结果到磁盘
    output_data = {
        'metrics': metrics,
        'results': results,
        'sampling_params': {
            'temperature': eval_sampling_params.temperature,
            'top_p': eval_sampling_params.top_p,
            'max_tokens': eval_sampling_params.max_tokens,
            'stop': eval_sampling_params.stop
        }
    }
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Evaluation completed!")
    print(f"Accuracy: {accuracy:.4f} ({correct_count}/{total_examples})")
    print(f"Format Accuracy: {format_accuracy:.4f} ({format_correct_count}/{total_examples})")
    print(f"Results saved to: {output_file}")
    
    return metrics


def analyze_response_format(response: str) -> Dict[str, bool]:
    """
    分析模型响应的格式是否符合要求
    
    Args:
        response: 模型生成的响应
        
    Returns:
        格式分析结果
    """
    has_think_tag = "<think>" in response and "</think>" in response
    has_answer_tag = "<answer>" in response and "</answer>" in response
    has_correct_structure = "</think> <answer>" in response
    
    return {
        'has_think_tag': has_think_tag,
        'has_answer_tag': has_answer_tag,
        'has_correct_structure': has_correct_structure,
        'is_fully_formatted': has_think_tag and has_answer_tag and has_correct_structure
    }


def main():
    """主函数：评估 Qwen 2.5 Math 1.5B 的零样本 MATH 性能"""
    
    # 配置路径
    math_validation_path = "./data/gsm8k/test.jsonl"
    output_path = "./qwen2.5_math_1.5b_zero_shot_math_results.json"
    
    # 检查文件是否存在
    if not os.path.exists(math_validation_path):
        raise FileNotFoundError(f"MATH validation file not found: {math_validation_path}")
    
    # 1. 加载 MATH 验证数据
    print("Loading MATH validation data...")
    math_examples = load_math_validation_data(math_validation_path)
    print(f"Loaded {len(math_examples)} examples")
    
    # 2. 格式化提示
    print("Formatting prompts...")
    prompts = []
    gold_answers = []
    
    for example in math_examples:
        # 使用 'question' 字段而不是 'problem'
        problem_text = example.get('question', '')
        answer_text = example.get('answer', '')
        
        # 从答案文本中提取最终答案
        final_answer = extract_final_answer(answer_text)
        
        formatted_prompt = format_r1_zero_prompt(problem_text)
        prompts.append(formatted_prompt)
        gold_answers.append(final_answer)
    
    # 打印一些示例用于调试
    print(f"\n示例提示: {prompts[0][:800]}...")
    print(f"\n示例提示: {prompts[1][:400]}...")
    print(f"\n示例提示: {prompts[2][:400]}...")
    print(f"示例答案: {gold_answers[0]}")
    print(f"示例答案: {gold_answers[1]}")
    print(f"示例答案: {gold_answers[2]}")

    
    # 3. 设置采样参数 - 确保模型能生成完整的格式
    sampling_params = SamplingParams(
        temperature=0.0,  # 对于评估，通常使用确定性生成
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],  # 确保模型在生成完答案后停止
        include_stop_str_in_output=True
    )
    
    # 4. 初始化 vLLM 模型
    print("Initializing vLLM model...")
    llm = LLM(model="Qwen/Qwen2.5-Math-1.5B-Instruct")
    
    
    # 5. 评估模型
    metrics = evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        gold_answers=gold_answers,
        eval_sampling_params=sampling_params,
        output_file=output_path
    )
    
    # 6. 打印详细结果
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Model: {metrics['model_name']}")
    print(f"Evaluation Type: {metrics['evaluation_type']}")
    print(f"Total Examples: {metrics['total_examples']}")
    print(f"Correct Answers: {metrics['correct_count']}")
    print(f"Format Correct: {metrics['format_correct_count']}")
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Format Accuracy: {metrics['format_accuracy']:.4f} ({metrics['format_accuracy']*100:.2f}%)")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()