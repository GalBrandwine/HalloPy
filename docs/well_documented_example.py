# -*- coding: utf-8 -*-
"""Demonstrate high quality docstrings.
Module-level docstrings appear as the first "statement" in a module. Remember,
that while strings are regular Python statements, comments are not, so an
inline comment may precede the module-level docstring.
After importing a module, you can access this special string object through the
``__doc__`` attribute; yes, it's actually available as a runtime attribute,
despite not being given an explicit name! The ``__doc__`` attribute is also
what is rendered when you call ``help()`` on a module, or really any other
object in Python.
You can also document a package using the module-level docstring in the
package's ``__init__.py`` file.
"""


def main():
    """Illustrate function-level docstring.
    Note that all docstrings begin with a one-line summary. The summary is
    written in the imperative mood ("do", "use", "find", "return", "render",
    etc) and ends with a period. The method signature is not, in any way,
    duplicated into the comments (that would be difficult to maintain).
    All subsequent paragraphs in a docstring are indented exactly the same as
    the summary line. The same applies to the closing quotation marks.
    """
    docs = Documentation()
    help(docs.__module__)


class Documentation(object):
    """Illustrate class-level docstring.
    Classes use a special whitespace convention: the opening and closing quotes
    are preceded and followed by a blank line, respectively. No other
    docstrings should be preceded or followed by anything but code.
    A blank line at the end of a multi-line docstring before the closing
    quotation marks simply makes it easier for tooling to auto-format
    paragraphs (wrapping them at 79 characters, per PEP8), without the closing
    quotation marks interfering. For example, in Vim, you can use `gqip` to
    "apply text formatting inside the paragraph." In Emacs, the equivalent
    would be the `fill-paragraph` command. While it's not required, the
    presence of a blank line is quite common and much appreciated. Regardless,
    the closing quotation marks should certainly be on a line by themselves.
    """

    def __init__(self):
        """Illustrate method-level docstring.
        All public callables should have docstrings, including magic methods
        like ``__init__()``.
        You'll notice that all these docstrings are wrapped in triple double
        quotes, as opposed to just "double quotes", 'single quotes', or
        '''triple single quotes.''' This is a convention for consistency and
        readability. However, there are two edge cases worth knowing about
        which I'll illustrate in just a moment.
        """
        super(Documentation, self).__init__()

    def oneliner(self):
        """Illustrate one line docstring, including the quotation marks."""
        return self.oneliner.__doc__

    def backslashes(self):
        r"""Illustrate raw triple double quotes for \backslashes\."""
        return self.backslashes.__doc__

    def __unicode__(self):
        u"""In Python 2, use Unicode triple double quotes for Ůňïčøđê."""
        return self.__unicode__.__doc__


if __name__ == '__main__':
    # No need for a docstring here, but an inline comment explaining that this
    # will only be executed when this module is run directly might be useful.
    # Try running this module!
    main()
