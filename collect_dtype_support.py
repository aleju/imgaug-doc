import os
import re
from collections import OrderedDict, defaultdict
import copy as copy_lib
import matplotlib.pyplot as plt
import matplotlib

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
BLACKLIST = {
    "/external/opensimplex.py",
    "/external/poly_point_isect_py2py3.py"
}
WRITE_TO = os.path.join(CURRENT_DIR, "images", "dtype_support", "%s.png")

EMPTY_LINE_PATTERN = re.compile(r"^[\s\t]*$")
CLASS_START_PATTERN = re.compile(r"^\s*class [a-zA-Z0-9_.(),\s]+:?[\s]*$")
FUNCTION_START_PATTERN = re.compile(r"^\s*def [a-zA-Z0-9_(),=\.\[\]{}:\-\"'\s\*]+:?[\s]*$")
HEADLINE_PATTERN = re.compile(r"^[\s\t]*-+[\s]*$")

DEFAULT_SCENARIO_NAME = "standard"
ELSE_SCENARIO_NAME = "else"


def main():
    print("Collecting filepaths to parse...")
    fps = get_filepaths_to_parse()
    print("Found %d filepaths: %s" % (len(fps), str(fps),))

    print("Parsing files...")
    dtype_supports = []
    for fp in fps:
        print("Parsing file '%s'..." % (fp,))
        dts = parse(fp)
        print("Found %d dtype support docstring blocks in the file." % (len(dts),))
        dtype_supports.extend(dts)
    print("Found a total of %d dtype support docstring blocks in all files." % (len(dtype_supports),))

    print("Plotting to file_pattern '%s'" % (WRITE_TO,))
    plot(dtype_supports, WRITE_TO)

    print("Finished.")


def get_filepaths_to_parse():
    result = []
    crawl_dir = os.path.join(CURRENT_DIR, "imgaug")
    for root, subdirs, files in os.walk(crawl_dir):
        for file in files:
            fp = os.path.join(root, file)
            if fp.endswith(".py") and file != "__init__.py" and fp[len(crawl_dir):] not in BLACKLIST:
                result.append(fp)
    return result


def parse(fp):
    dtype_supports = []

    with open(fp, "r") as f:
        content = [line.rstrip() for line in f.readlines()]

    for i in range(len(content)):
        line = content[i]
        if "#" in line:
            pos = line.index("#")
            content[i] = line[:pos]

    group = rstripword(fp[len(CURRENT_DIR):], ".py").lstrip("/").replace("/", ".")
    current_type = ""
    current_name = ""
    is_waiting_for_docstring = False
    is_in_docstring = False
    docstring_buffer = []
    i = 0
    while i < len(content):
        line = content[i]
        is_line_empty = EMPTY_LINE_PATTERN.match(line) is not None
        is_line_docstring_start_or_end = '"""' in line
        is_line_class_start = CLASS_START_PATTERN.match(line) is not None
        is_line_function_start = FUNCTION_START_PATTERN.match(line) is not None

        if is_line_docstring_start_or_end:
            is_single_line_docstring = line.strip().startswith('"""') and line.strip().endswith('"""') and len(line.strip()) >= 6
            if is_in_docstring or is_single_line_docstring:
                # handling here
                docstring_buffer.append(line)

                opsup = DtypeSupportForOperation.from_docstring(
                    current_type, group, current_name,
                    "\n".join(docstring_buffer))
                if opsup is not None:
                    dtype_supports.append(opsup)

                # reset
                is_waiting_for_docstring = False
                is_in_docstring = False
                current_type = ""
                current_name = ""
                docstring_buffer = []
            elif is_waiting_for_docstring:
                is_in_docstring = True
                is_waiting_for_docstring = False
                docstring_buffer.append(line)
            else:
                pass
        elif is_line_class_start or is_line_function_start:
            is_waiting_for_docstring = True
            current_type = "class" if is_line_class_start else "function"
            current_name = line.strip()
            current_name = lstripword(current_name, "def ")
            current_name = lstripword(current_name, "class ")
            current_name = current_name.rstrip("):")
            if "(" in current_name:
                current_name = current_name[:current_name.index("(")]
            c = 0
            found = False
            while not found and c < 3:
                if not content[i].strip().endswith(":"):
                    i += 1
                else:
                    found = True
            if not found:
                is_waiting_for_docstring = False
                current_type = ""
                current_name = ""
            i = i - c
        elif is_in_docstring:
            docstring_buffer.append(line)
        elif not is_line_empty and is_waiting_for_docstring:
            is_waiting_for_docstring = False

        i += 1

    return dtype_supports


def plot(dtype_supports, save_fp_pattern):
    """
    for dts in dtype_supports:
        pointer = dts.group + "." + dts.name
        for scenario in dts.scenarios:
            print("-----------")
            print("%s (%s):" % (pointer, scenario.name))
            for cell in scenario.grid.resolve(dtype_supports).cells:
                print("* %s: %s; %s %s" % (cell.dtype, cell.support_level, cell.test_level, str(cell.comments)))
                pass
    """
    bg_color_mapping = {
        DtypeSupportGridCell.SUPPORT_LEVEL_YES: "#00AA00",
        DtypeSupportGridCell.SUPPORT_LEVEL_LIMITED: "#AAAA00",
        DtypeSupportGridCell.SUPPORT_LEVEL_NO: "#AA0000",
        DtypeSupportGridCell.SUPPORT_LEVEL_UNKNOWN: "#FFFFFF",
    }
    test_level_abbreviations = {
        DtypeSupportGridCell.TEST_LEVEL_FULLY_TESTED: "+++",
        DtypeSupportGridCell.TEST_LEVEL_TESTED: "++",
        DtypeSupportGridCell.TEST_LEVEL_INDIRECTLY_TESTED: "+",
        DtypeSupportGridCell.TEST_LEVEL_NOT_TESTED: "-",
    }

    dts_by_groups = defaultdict(list)
    for dts in dtype_supports:
        dts_by_groups[dts.group].append(dts)

    for group in dts_by_groups:
        columns = ('uint8', 'uint16', 'uint32', 'uint64',
                   'int8', 'int16', 'int32', 'int64',
                   'float16', 'float32', 'float64', 'float128',
                   'bool')
        rows = []
        data_list = []
        cell_bg_colors = []

        for dts in dts_by_groups[group]:
            # skip private functions/methods/classes
            if dts.name.startswith("_"):
                continue

            for scenario in dts.scenarios:
                if len(dts.scenarios) == 1:
                    rows.append("%s" % (dts.name,))
                else:
                    rows.append("%s (%s)" % (dts.name, scenario.name))

                grid = scenario.grid.resolve(dtype_supports)
                test_levels = [grid.get_cell_by_dtype(col).test_level for col in columns]
                support_levels = [grid.get_cell_by_dtype(col).support_level for col in columns]
                data_list_row = []
                for tl, sl in zip(test_levels, support_levels):
                    if sl == DtypeSupportGridCell.SUPPORT_LEVEL_NO:
                        data_list_row.append("")
                    elif sl == DtypeSupportGridCell.SUPPORT_LEVEL_UNKNOWN:
                        data_list_row.append("?")
                    else:
                        data_list_row.append(test_level_abbreviations[tl])
                data_list.append(data_list_row)
                cell_bg_colors.append([bg_color_mapping[sl] for sl in support_levels])

        fig = plt.figure(1)
        fig.tight_layout()

        ax = plt.subplot2grid((1, 1), (0, 0), colspan=1, rowspan=1)
        ax.set_title(group)
        table = ax.table(cellText=data_list,
                         rowLabels=rows,
                         colLabels=columns,
                         cellColours=cell_bg_colors,
                         loc="upper center")

        ax.axis("off")
        ax.grid(False)

        fig.set_size_inches(w=12, h=5+int(0.5*len(dts_by_groups[group])))

        plt.gcf().canvas.draw()
        points = table.get_window_extent(plt.gcf()._cachedRenderer).get_points()
        points[0, :] -= 10
        points[1, :] += 10
        nbbox = matplotlib.transforms.Bbox.from_extents(points / plt.gcf().dpi)

        fig.savefig(save_fp_pattern % (group.replace(".", "_"),), bbox_inches=nbbox)
        plt.close()


def count_inset(line):
    return len(line) - len(line.lstrip(" "))


def lstripword(s, word):
    if s.startswith(word):
        return s[len(word):]
    return s


def rstripword(s, word):
    if s.endswith(word):
        return s[:-len(word)]
    return s


class DtypeSupportForOperation(object):
    OP_TYPE_CLASS = "class"
    OP_TYPE_FUNCTION = "function"

    def __init__(self, op_type, group, name, scenarios):
        self.op_type = op_type
        self.group = group
        self.name = name
        self.scenarios = scenarios

    @classmethod
    def from_docstring(cls, op_type, group, name, docstring):
        # remove '"""' and whitespace-likes at the end
        docstring = docstring.rstrip('"').rstrip()

        docstring = docstring.split("\n")

        # read out only part below "dtype support::"
        in_dtype_block = False
        dtype_block = []
        for line in docstring:
            if in_dtype_block:
                dtype_block.append(line)
            if "**Supported dtypes**:" in line:
                in_dtype_block = True

        if len(dtype_block) == 0:
            return None

        dtype_block = "\n".join(dtype_block).strip("\n").split("\n")
        dtype_block = [line.strip("\n") for line in dtype_block]

        # split into subblocks by inset level
        subblocks = []
        current_subblock = []
        last_inset = count_inset(dtype_block[0])
        for line in dtype_block:
            current_inset = count_inset(line)
            if len(line.strip()) > 0 and current_inset != last_inset:
                subblocks.append((last_inset, current_subblock))
                current_subblock = []
                last_inset = count_inset(line)
            current_subblock.append(line)
        if len(current_subblock) > 0:
            subblocks.append((last_inset, current_subblock))

        # remove everything starting with the next headline
        next_headline = -1
        for block_idx, (inset_level, block) in enumerate(subblocks):
            for line_idx, line in enumerate(block[1:]):
                if HEADLINE_PATTERN.match(line):
                    block[:] = block[:line_idx-1]
                    next_headline = block_idx
                    break

            if next_headline > -1:
                break

        if next_headline > -1:
            # remove until (and including) block that contains headline
            subblocks = subblocks[:next_headline+1]
            # is block containing headline now empty? then delete that block
            if next_headline > 0:
                inset_level, block = subblocks[next_headline]
                if len(block) == 0:
                    subblocks = subblocks[:next_headline]

        # convert subblocks of (inset, content) to (scenario name, content)
        scenario_to_support = []
        i = 0
        while i < len(subblocks):
            subblock_inset = subblocks[i][0]
            subblock_content = "\n".join(subblocks[i][1])

            is_not_empty = len(subblock_content) > 0
            is_starting_with_if = subblock_content.strip().lower().startswith("if")
            is_starting_with_else = subblock_content.strip().lower().startswith("else")
            is_ending_with_colon = subblock_content.strip().endswith(":")
            is_grid_content = "* ``uint8``" in subblock_content.strip()
            is_see = (
                ("See ``" in subblock_content.strip())
                or ("See :func:`" in subblock_content.strip())
                or ("See :class:`" in subblock_content.strip())
            )
            is_minimum_of = "minimum of (" in subblock_content.strip()

            child_blocks = []
            n_skip_children = 0
            for j in range(i + 1, len(subblocks)):
                lower_inset_level = subblocks[j][0] > subblock_inset
                same_inset_level = subblocks[j][0] == subblock_inset
                last_line_was_empty = child_blocks[-1].endswith("\n") if child_blocks else False
                if lower_inset_level or (same_inset_level and not last_line_was_empty):
                    child_blocks.append("\n".join(subblocks[j][1]))
                    n_skip_children += 1
                else:
                    break

            if is_not_empty and (is_starting_with_if or is_starting_with_else) and is_ending_with_colon:
                if is_starting_with_else:
                    scenario_name = ELSE_SCENARIO_NAME
                else:
                    scenario_name = subblock_content.strip()
                    scenario_name = lstripword(scenario_name, "If (")
                    scenario_name = lstripword(scenario_name, "if (")
                    scenario_name = scenario_name.rstrip("):")
                scenario_to_support.append((scenario_name, child_blocks))
                i += n_skip_children
            elif is_not_empty and (is_grid_content or is_see or is_minimum_of):
                scenario_name = DEFAULT_SCENARIO_NAME
                scenario_to_support.append((scenario_name, subblocks[i][1] + child_blocks))
                i += n_skip_children

            i += 1

        # parse scenarios
        scenarios = []
        for scenario_descr, childblocks in scenario_to_support:
            childblocks_str = "\n".join([
                "\n".join(childblock) if isinstance(childblock, list) else childblock for childblock in childblocks
            ])

            grid = cls._parse_scenario_to_grid(childblocks_str)
            scenarios.append(DtypeSupportScenario(scenario_descr, grid))

        return DtypeSupportForOperation(op_type, group, name, scenarios)

    @classmethod
    def _parse_scenario_to_grid(cls, docstring):
        # Remove comments from scenario docstring, if there are any.
        # this is important, because the comments may also contain formulations
        # like "See XYZ".
        docstring_no_comments = re.sub("^[\n\s\t]+", "", docstring)
        docstring_no_comments = docstring_no_comments.split("\n\n")[0]

        if ("See :func:`" in docstring_no_comments
                or "See :class:`" in docstring_no_comments
                or "See ``" in docstring_no_comments):
            return cls._parse_scenario_to_grid_see(docstring)
        elif "minimum of (" in docstring:
            return cls._parse_scenario_to_grid_minimum(docstring)
        else:
            return cls._parse_scenario_to_grid_standard(docstring)

    @classmethod
    def _parse_scenario_to_grid_see(cls, docstring):
        docstring = docstring.strip()
        if ":func:" in docstring:
            op_type = "function"
            match = re.match(r"^[\s\t]*See :func:`~?(?P<name>[^`]+)`.*$", docstring.strip())
        elif ":class:" in docstring:
            op_type = "class"
            match = re.match(r"^[\s\t]*See :class:`~?(?P<name>[^`]+)`.*$", docstring.strip())
        else:
            op_type = "class"
            match = re.match(r"^[\s\t]*See ``~?(?P<name>[^`]+)``.*$", docstring.strip())

        if match is None:
            msg = (
                "Tried to parse a 'See <pointer>' string, but found neither a "
                "function indicator, nor a class indicator. Docstring: %s" % (
                    docstring))
            raise ValueError(msg)

        op_name = match.groupdict()["name"]
        scenario_name = None
        if "(" in op_name:
            pos = op_name.index("(")
            scenario_name = op_name[pos+1:-1]
            op_name = op_name[:pos]
        return DtypeSupportGridSee(op_type, op_name, scenario_name)

    @classmethod
    def _parse_scenario_to_grid_minimum(cls, docstring):
        docstring = "\n".join(docstring.strip().split("\n")[1:-1])
        children = []
        for line in docstring.split("\n"):
            line = line.strip()
            line = line.rstrip(",")
            if ":func:" in line:
                op_type = "function"
                match = re.match(r"^[\s\t]*:func:`~?(?P<name>[^`]+)`.*$", line)
            elif ":class:" in line:
                op_type = "class"
                match = re.match(r"^[\s\t]*:class:`~?(?P<name>[^`]+)`.*$", line)
            else:
                op_type = "class"
                match = re.match(r"^[\s\t]*``(?P<name>[^`]+)``.*$", line)

            op_name = match.groupdict()["name"]
            scenario_name = None
            if "(" in op_name:
                pos = op_name.index("(")
                scenario_name = op_name[pos+1:-1]
                op_name = op_name[:pos]

            children.append(DtypeSupportGridSee(op_type, op_name, scenario_name))
        return DtypeSupportGridMinimum(children)

    @classmethod
    def _parse_scenario_to_grid_standard(cls, docstring):
        assert "* ``uint8``" in docstring
        docstring_pieces = re.split(r"[\n]{2,}", docstring)
        support = docstring_pieces[0].strip()
        comments = docstring_pieces[1].strip() if len(docstring_pieces) > 1 else ""

        supports_parsed = []
        for line in support.split("\n"):
            support_line_pieces_pattern = re.compile(
                r"^[\s\t]*?\* ``(?P<dtype>[a-z0-9]+)``: "
                + r"(?P<support_level>yes|limited|no|\?)"
                + r"(?:; (?P<test_level>not tested|indirectly tested|tested|fully tested))?\s*"
                + r"(?P<comment_numbers>[\s()0-9]+)?[\s\t\n]*?$"
            )
            matches = support_line_pieces_pattern.match(line)

            if matches:
                gdict = matches.groupdict()
                dtype = gdict["dtype"]
                support_level = gdict["support_level"]
                test_level = gdict["test_level"]
                test_level = test_level if test_level is not None else ""
                comment_numbers = gdict["comment_numbers"]
                comment_numbers = comment_numbers if comment_numbers is not None else ""
                if comment_numbers:
                    comment_numbers = [int(piece.strip()[1:-1]) for piece in re.split(r"\s+", comment_numbers)]
                else:
                    comment_numbers = []
                supports_parsed.append({
                    "dtype": dtype.strip(),
                    "support_level": support_level.strip(),
                    "test_level": test_level.strip(),
                    "comment_numbers": comment_numbers
                })

        comments_parsed = {}
        # remove anything except comment blocks
        comments = [comments_i
                    for comments_i
                    in re.split(r"\n{2,}", comments)
                    if re.match(r"^\s*?- \([0-9]+\).*$", comments_i.replace("\n", " "))]
        for comments_i in comments:
            comments_split_pattern = re.compile(r"(?:^\s*?|\s+)- ")
            comment_extract_pattern = re.compile(r"\((?P<number>[0-9]+)\) (?P<text>.*)")
            bullet_points = comments_split_pattern.split(comments_i.replace("\n", " "))
            for bullet_point in bullet_points:
                comment_pieces = comment_extract_pattern.match(bullet_point)
                if comment_pieces is not None:
                    number = comment_pieces.groupdict()["number"]
                    text = comment_pieces.groupdict()["text"]
                    text = re.sub(r"\s+", " ", text)
                    comments_parsed[int(number.strip())] = text.strip()

        cells = []
        for support_parsed in supports_parsed:
            cells.append(
                DtypeSupportGridCell(
                    dtype=support_parsed["dtype"],
                    support_level=support_parsed["support_level"],
                    test_level=support_parsed["test_level"],
                    comments=[comments_parsed[number] for number in support_parsed["comment_numbers"]]
                )
            )

        return DtypeSupportGrid(cells)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "DtypeSupportForOperation(op_type=%s, group=%s, name=%s, scenarios=%s)" % (
            self.op_type,
            self.group,
            self.name,
            self.scenarios
        )


class DtypeSupportScenario(object):
    def __init__(self, name, grid):
        self.name = name
        self.grid = grid

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "DtypeSupportScenario(name=%s, grid=%s)" % (
            self.name,
            self.grid
        )


class DtypeSupportGridSee(object):
    def __init__(self, other_op_type, other_pointer, other_scenario):
        self.other_op_type = other_op_type
        self.other_pointer = other_pointer
        self.other_scenario = other_scenario

    def resolve(self, other_dtype_supports):
        for dtsup in other_dtype_supports:
            same_name = ((dtsup.group + "." + dtsup.name) == self.other_pointer)
            if same_name:  # ignore op type here, because some functions hide as classes
                if self.other_scenario is None:
                    return self._pick_grid_of_minimal_support_scenario(dtsup.scenarios, other_dtype_supports)
                else:
                    for scenario in dtsup.scenarios:
                        if scenario.name == self.other_scenario:
                            return scenario.grid.resolve(other_dtype_supports)
        raise Exception("Did not find op type=%s, op pointer=%s, scenario name=%s" % (
            self.other_op_type, self.other_pointer, self.other_scenario))

    @classmethod
    def _pick_grid_of_minimal_support_scenario(cls, scenarios, other_dtype_supports):
        if len(scenarios) == 1:
            return scenarios[0].grid.resolve(other_dtype_supports)
        assert len(scenarios) > 0
        grid_min = None
        for i, scenario in enumerate(scenarios):
            grid = scenario.grid.resolve(other_dtype_supports)
            if grid_min is None or grid.has_lower_support_than(grid_min):
                grid_min = grid

        return grid_min

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "DtypeSupportGridSee(other_op_type=%s, other_pointer=%s, other_scenario=%s)" % (
            self.other_op_type,
            self.other_pointer,
            self.other_scenario
        )


class DtypeSupportGridMinimum(object):
    def __init__(self, children):
        self.children = children

    def resolve(self, other_dtype_support):
        grid = None
        for child in self.children:
            child_r = child.resolve(other_dtype_support)
            if grid is None:
                grid = child_r
            else:
                grid = self._generate_minimum_grid(grid, child_r)
        return grid

    @classmethod
    def _generate_minimum_grid(cls, grid_a, grid_b):
        dtype_to_cell = {}
        for cell in grid_a.cells:
            dtype_to_cell[cell.dtype] = copy_lib.deepcopy(cell)
        for cell in grid_b.cells:
            if cell.dtype not in dtype_to_cell:
                dtype_to_cell[cell.dtype] = DtypeSupportGridCell(
                    dtype=cell.dtype,
                    support_level=DtypeSupportGridCell.SUPPORT_LEVEL_NO,
                    test_level=DtypeSupportGridCell.TEST_LEVEL_NOT_TESTED,
                    comments=cell.comments
                )
            else:
                current = dtype_to_cell[cell.dtype]
                dtype_to_cell[cell.dtype] = DtypeSupportGridCell(
                    dtype=cell.dtype,
                    support_level=DtypeSupportGridCell.min_support_level([current.support_level, cell.support_level]),
                    test_level=DtypeSupportGridCell.min_test_level([current.test_level, cell.test_level]),
                    comments=current.comments + cell.comments
                )

        cells = list(dtype_to_cell.values())
        grid = DtypeSupportGrid(cells)
        return grid

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "DtypeSupportGridMinimum(%s)" % (
            str([str(child) for child in self.children])
        )


class DtypeSupportGrid(object):
    def __init__(self, cells):
        order = ["uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64",
                 "float16", "float32", "float64", "float128", "bool"]
        order_dict = OrderedDict()
        for dt in order:
            order_dict[dt] = None

        for cell in cells:
            assert cell.dtype in order_dict
            order_dict[cell.dtype] = cell
        cells_ordered = list(order_dict.values())
        all_found = all([v is not None for v in cells_ordered])
        if not all_found:
            dt_names = [key for key, value in cells_ordered.items()
                        if value is None]
            msg = "Could not find the following dtypes: %s" % (
                ", ".join(dt_names),)
            raise ValueError(msg)
        self.cells = cells_ordered

    def add(self, dtype, support_level, test_level, comments):
        self.cells.append(DtypeSupportGridCell(dtype, support_level, test_level, comments))

    def resolve(self, other_dtype_supports):
        return self

    def get_cell_by_dtype(self, dtype):
        for cell in self.cells:
            if cell.dtype == dtype:
                return cell
        raise Exception("Did not found cell with dtype '%s'" % (dtype,))

    def has_lower_support_than(self, other_grid):
        nb_lower = 0
        for cell in self.cells:
            other_cell = other_grid.get_cell_by_dtype(cell.dtype)
            sl = cell.support_level
            other_sl = other_cell.support_level
            sl_min = DtypeSupportGridCell.min_support_level([sl, other_sl])
            if sl == other_sl:
                pass
            elif sl_min == sl:
                nb_lower += 1
            elif sl_min == other_sl:
                nb_lower -= 1
        return nb_lower > 0

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "DtypeSupportGrid(%s)" % (
            str([str(cell) for cell in self.cells])
        )

    def to_pretty_string(self):
        s = []
        for cell in self.cells:
            s.append("* %s" % (cell.to_pretty_string(),))
        return "\n".join(s)


class DtypeSupportGridCell(object):
    SUPPORT_LEVEL_YES = "yes"
    SUPPORT_LEVEL_LIMITED = "limited"
    SUPPORT_LEVEL_NO = "no"
    SUPPORT_LEVEL_UNKNOWN = "?"

    TEST_LEVEL_FULLY_TESTED = "fully tested"
    TEST_LEVEL_TESTED = "tested"
    TEST_LEVEL_INDIRECTLY_TESTED = "indirectly tested"
    TEST_LEVEL_NOT_TESTED = "not tested"

    def __init__(self, dtype, support_level, test_level, comments):
        assert support_level in [self.SUPPORT_LEVEL_YES, self.SUPPORT_LEVEL_LIMITED, self.SUPPORT_LEVEL_NO,
                                 self.SUPPORT_LEVEL_UNKNOWN]
        if test_level == "":
            test_level = self.TEST_LEVEL_NOT_TESTED
        assert test_level in [self.TEST_LEVEL_FULLY_TESTED, self.TEST_LEVEL_INDIRECTLY_TESTED,
                              self.TEST_LEVEL_TESTED, self.TEST_LEVEL_NOT_TESTED]
        self.dtype = dtype
        self.support_level = support_level
        self.test_level = test_level
        self.comments = comments

    @classmethod
    def min_support_level(cls, support_levels):
        if len(support_levels) == 0:
            raise Exception("Cant estimate minimum support level of nothing.")
        current = cls.SUPPORT_LEVEL_YES
        for level in support_levels:
            if cls.is_lower_support_level(level, current):
                current = level
        return current

    @classmethod
    def min_test_level(cls, test_levels):
        if len(test_levels) == 0:
            raise Exception("Cant estimate minimum test level of nothing.")
        current = cls.TEST_LEVEL_FULLY_TESTED
        for level in test_levels:
            if cls.is_lower_test_level(level, current):
                current = level
        return current

    @classmethod
    def is_lower_support_level(cls, this, than_that):
        mapping = {
            cls.SUPPORT_LEVEL_YES: 3,
            cls.SUPPORT_LEVEL_LIMITED: 2,
            cls.SUPPORT_LEVEL_UNKNOWN: 1,
            cls.SUPPORT_LEVEL_NO: 0
        }
        return mapping[this] < mapping[than_that]

    @classmethod
    def is_lower_test_level(cls, this, than_that):
        mapping = {
            cls.TEST_LEVEL_FULLY_TESTED: 3,
            cls.TEST_LEVEL_TESTED: 2,
            cls.TEST_LEVEL_INDIRECTLY_TESTED: 1,
            cls.TEST_LEVEL_NOT_TESTED: 0
        }
        return mapping[this] < mapping[than_that]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "DtypeSupportGridCell(dtype=%s, support_level=%s, test_level=%s, comments=%s)" % (
            self.dtype,
            self.support_level,
            self.test_level,
            str(self.comments)
        )

    def to_pretty_string(self):
        return "%s: %s; %s (%d cmts)" % (
            self.dtype,
            self.support_level,
            self.test_level,
            len(self.comments)
        )


if __name__ == "__main__":
    main()
