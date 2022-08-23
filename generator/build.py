from ast import AST
import sys 
import argparse
import functools
import re
import os
import shutil
import logging
import yaml
from distutils.dir_util import copy_tree

import dominate
from dominate import document
from dominate.tags import *
from dominate.util import raw
import mistletoe
from mistletoe.ast_renderer import ASTRenderer 


parser = argparse.ArgumentParser(description="Build a blog post.")

parser.add_argument(
  '--src',
  default='',
  type=str,
  help='The source folder to use for building the blog post. This folder should contain at minimum a build.yml file, and a markdown file named the same as its parent folder.'
)
parser.add_argument(
  '-r',
  action='store_true'
)
parser.add_argument(
  '--dest',
  default='build',
  type=str,
  help='The destination directory for all built content.'
)

args = parser.parse_args()

class Parser:
  def __init__(self):
    self.tags = {}

    self.document = document(
      title="Test",
      request=""
    )

    # are we in a code block?
    self.in_code = False

    self.content = div(id='content')
    self.document.body += self.content 

    self.parse_raw = False

    self.elements = [self.document, self.document.body, self.content]

    self.id = 0

  def gen_id(self, prefix: str = ''):
    self.id += 1
    return prefix + str(self.id)
  
  def parse(self, filename):
    with open(filename, 'r') as f:
      lines = f.readlines()
      lines = [l.replace('\n', '').replace('\r', '') for l in lines]

      for idx in range (len(lines)):
        line = lines[idx]

        # ignore blank lines, unless we're in a code block
        if not line and not self.in_code:
          continue 

        for t in self.tags:
          if line.startswith(t + ' ') or line==t:
            self.tags[t](self, line)
            break; 
        else:
          self.tags['default'](self, line)

      return self.document
    
  def process_for(self, prefix: str, remove_prefix: bool = True):
    def wrapper(func):
      @functools.wraps(func)
      def wrapped(parser, content, *args, **kwargs):
        if remove_prefix:
          # remove the prefix from the content
          content = content[len(prefix) + 1:]

        return func(parser, content, *args, **kwargs)

      self.tags[prefix] = wrapped
      return wrapped
    
    return wrapper

parser = Parser()

@parser.process_for('#')
def generate_title(parser: Parser, content: str):
  parser.elements[-1] += div(content, cls='title')

@parser.process_for('##')
def generate_subtitle(parser: Parser, content: str):
  parser.elements[-1] += div(content, cls='section')

@parser.process_for('###')
def generate_subtitle(parser: Parser, content: str):
  parser.elements[-1] += div(content, cls='subsection')

@parser.process_for('Date:')
def generate_date(parser: Parser, content: str):
  parser.elements[-1] += div(content, cls='date')

@parser.process_for('Intro:')
def generate_intro(parser: Parser, content: str):
  parser.elements[-1] += div(raw(mistletoe.markdown(content)), cls='section intro')

@parser.process_for(':Head:')
def start_head(parser: Parser, content: str):
  parser.parse_raw = False
  parser.elements.append(parser.document.head)

@parser.process_for('-Head-')
def end_head(parser: Parser, content: str):
  parser.parse_raw = True
  del parser.elements[-1]

@parser.process_for(':Equation:')
def start_equation(parser: Parser, content: str):
  parser.parse_raw = False
  equation_element = div(content, cls='eq')
  parser.elements[-1] += equation_element
  parser.elements.append(equation_element) 

@parser.process_for('-Equation-')
def end_euqation(parser: Parser, content: str):
  parser.parse_raw = True
  del parser.elements[-1]

@parser.process_for(':Raw:')
def start_raw(parser: Parser, content: str):
  parser.parse_raw = False

@parser.process_for('-Raw-')
def end_raw(parser: Parser, content: str):
  parser.parse_raw = True 

@parser.process_for('```')
def start_end_code(parser: Parser, content: str):
  if parser.in_code:
    # when parsing under the "defaulr" rule
    # text has a newline prepended to it
    # for code, we would like to remove this newline at the 
    # beginning
    parser.elements[-1][0] = str(parser.elements[-1][0])[1:] 

    del parser.elements[-1]
    parser.in_code = False 
    parser.parse_raw = True
  else:
    language = content
    if not language: language = 'plaintext'

    pre_element = pre()
    code_element = code(cls=f'language-{language}')

    pre_element += code_element
    parser.elements[-1] += pre_element

    parser.elements.append(code_element)

    parser.in_code = True
    parser.parse_raw = False

@parser.process_for("Image:", remove_prefix=False)
def generate_image(parser: Parser, content: str):
  m = re.search('!\[[^\]]*\]\((.*?)\s*("(?:.*[^"])")?\s*\)', content)

  alt_text = m.group(0)
  filename = m.group(1)
  title = m.group(2)[1:-1] 

  # get sibling 
  older_sibling = parser.elements[-1][-1]
  img_id = parser.gen_id('p-')
  older_sibling['id'] = img_id 

  image_element = div(cls='img')

  image_element += img(src=filename, alt=alt_text, title=title)
  parser.elements[-1] += image_element

  image_element += span(raw(title), cls='img-description')
  image_element['assign_to'] = img_id

@parser.process_for('default', remove_prefix=False)
def add_raw(parser: Parser, content: str):
  if parser.parse_raw:
    content = re.sub(r'\$(.*?)\$', r'\\\(\1\\\)', content)
    content = mistletoe.markdown(content)
  
  content = raw('\n' + content)

  if parser.parse_raw:
    content = div(content)

  parser.elements[-1] += content

@parser.process_for('Equation:')
def generate_equation(parser: Parser, content: str):
  div_element = div(f"\({content}\)", cls='eq')
  parser.elements[-1] += div_element 

def load_yaml(filename: str) -> dict:
  with open(filename, 'r') as f:
    config = yaml.safe_load(f)
  
  return config

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)

  # delete and remake the build folder
  dest = args.dest
  dest_path = os.path.join('./', dest)

  try:
    shutil.rmtree(dest_path)
  except FileNotFoundError:
    logging.info("Could not find build folder, creating a new one...")

  os.mkdir(dest_path)


  def build_dir(src):
    src_path = os.path.join('./', src)

    # copy everything in here to the build folder
    copy_dest_path = os.path.join(dest_path, src)
    if 'build.yml' in os.listdir(src_path):
      config = load_yaml(os.path.join(src_path, 'build.yml'))

      if config is None: config = {}

      if 'dest_path' in config:
        copy_dest_path = os.path.join(dest_path, config['dest_path'])

      logging.info("Copying %s to %s" % (src_path, copy_dest_path))
      copy_tree(src_path, copy_dest_path)

      # immediately delete the build.yml file
      os.remove(os.path.join(copy_dest_path, 'build.yml'))

      if 'copy_include' not in config or\
         ('copy_include' in config and config['copy_include']):
        # copy contents of include into the folder
        include_path = os.path.join(
          os.path.dirname(os.path.realpath(__file__)),
          'include'
        )

        logging.info("Copying template files...")
        copy_tree(include_path, os.path.join(copy_dest_path, 'include'))

      directory_name = os.path.basename(copy_dest_path) 

      # look for file with name "{directory_name}.md"
      index_markdown_file = f"{directory_name}.md"
      if index_markdown_file in os.listdir(copy_dest_path):
        logging.info(f"Found {index_markdown_file}, building...")

        filename = os.path.join(copy_dest_path, index_markdown_file)
        dest_filename = os.path.join(copy_dest_path, 'index.html')
        with open(dest_filename, 'w') as f:
          f.write(parser.parse(filename).render())
        
        # delete the markdown file
        os.remove(filename)

  if args.src != '':
    build_dir(args.src)

  else:
    for d in os.listdir():
      try:
        build_dir(d)
      except NotADirectoryError:
        logging.info(f"{d} is not a directory, skipping...")

  """
  with open('build/index.html', 'w') as f:
    f.write(parser.parse(filename).render())
  """

    

  