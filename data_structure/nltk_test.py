# https://likegeeks.com/nlp-tutorial-using-python-nltk/

# install nltk: $pip install nltk
# import nltk

# download nltk package: nltk.download('package_name')

# text tokenization
import urllib.request

import nltk

response = urllib.request.urlopen('http://php.net/')

html = response.read()

print(html)

# <!DOCTYPE html>\n<html xmlns="http://www.w3.org/1999/xhtml" lang="en">\n<head>\n\n  <meta charset="utf-8">\n  <meta name="viewport" content="width=dev

# We can use BeautifulSoup to clean the grabbed text like this:
from bs4 import BeautifulSoup

import urllib.request

response = urllib.request.urlopen('http://php.net/')

html = response.read()

soup = BeautifulSoup(html)

text = soup.get_text(strip=True)

print(text)

# PHP: Hypertext PreprocessorDownloadsDocumentationGet InvolvedHelpGetting StartedIntroductionA simple tutorialLanguage ReferenceBasic syntaxTypesVariablesConstantsExpressionsOperatorsControl StructuresFunctionsClasses and ObjectsNamespacesErrorsExceptionsGeneratorsReferences ExplainedPredefined VariablesPredefined ExceptionsPredefined Interfaces and ClassesContext options and parametersSupported Protocols and WrappersSecurityIntroductionGeneral considerationsInstalled as CGI binaryInstalled as an Apache moduleSession SecurityFilesystem SecurityDatabase SecurityError ReportingUsing Register GlobalsUser Submitted DataMagic QuotesHiding PHPKeeping CurrentFeaturesHTTP authentication with PHPCookiesSessionsDealing with XFormsHandling file uploadsUsing remote filesConnection handlingPersistent Database ConnectionsSafe ModeCommand line usageGarbage CollectionDTrace Dynamic TracingFunction ReferenceAffecting PHP's BehaviourAudio Formats ManipulationAuthentication ServicesCommand Line Specific ExtensionsCompression and Archive ExtensionsCredit Card ProcessingCryptography ExtensionsDatabase ExtensionsDate and Time Related ExtensionsFile System Related ExtensionsHuman Language and Character Encoding SupportImage Processing and GenerationMail Related ExtensionsMathematical ExtensionsNon-Text MIME OutputProcess Control ExtensionsOther Basic ExtensionsOther ServicesSearch Engine ExtensionsServer Specific ExtensionsSession ExtensionsText ProcessingVariable and Type Related ExtensionsWeb ServicesWindows Only ExtensionsXML ManipulationGUI ExtensionsKeyboard Shortcuts?This helpjNext menu itemkPrevious menu itemg pPrevious man pageg nNext man pageGScroll to bottomg gScroll to topg hGoto homepageg sGoto search(current page)/Focus search boxPHP is a popular general-purpose scripting language that is especially suited to web development.Fast, flexible and pragmatic, PHP powers everything from your blog to the most popular websites in the world.Download5.6.38·Release Notes·Upgrading7.0.32·Release Notes·Upgrading7.1.23·Release Notes·Upgrading7.2.11·Release Notes·Upgrading25 Oct 2018PHP 7.3.0RC4 ReleasedThe PHP team is glad to announce the next PHP 7.3.0 pre-release, PHP 7.3.0RC4.
# The rough outline of the PHP 7.3 release cycle is specified in thePHP Wiki.

# now convert text to tokens
tokens = [t for t in text.split()]

print(tokens)

# word based tokens
# ['PHP:', 'Hypertext', 'PreprocessorDownloadsDocumentationGet', 'InvolvedHelpGetting', 'StartedIntroductionA', 'simple']



freq = nltk.FreqDist(tokens)

for key, val in freq.items():
    print(str(key) + ':' + str(val))

# outline:11
# of:92
# 7.3:11
# release:63
# cycle:10
# specified:11
# thePHP:11
# Wiki.For:11
# source:30

freq.plot(20, cumulative=False)

# removing stopwords

from nltk.corpus import stopwords

stopwords.words('english')

clean_tokens = []

sr = stopwords.words('english')

for token in tokens:
    if token not in stopwords.words('english'):
        clean_tokens.append(token)

print(clean_tokens)

freq = nltk.FreqDist(clean_tokens)
freq.plot(20, cumulative=False)

# Tokenize Text Using NLTK

# To tokenize this text to sentences, we will use sentence tokenizer:
from nltk.tokenize import sent_tokenize

mytext = "Hello Adam, how are you? I hope everything is going well. Today is a good day, see you dude."

print(sent_tokenize(mytext))
# ['Hello Adam, how are you?', 'I hope everything is going well.', 'Today is a good day, see you dude.']

# word tokenizer
from nltk.tokenize import word_tokenize

mytext = "Hello Mr. Adam, how are you? I hope everything is going well. Today is a good day, see you dude."

print(word_tokenize(mytext))

# Get Synonyms from WordNet(Dictionary)

from nltk.corpus import wordnet

synonyms = wordnet.synsets("pain")

print(synonyms[0].definition())

print(synonyms[0].examples())

# a symptom of some physical hurt or disorder

# ['the patient developed severe pain and distension']

# stemming
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

print(stemmer.stem('working'))
# work

# lematizing
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize('increases'))
# increase

# verb, noun,adjective,adverb
print(lemmatizer.lemmatize('playing', pos="v"))

print(lemmatizer.lemmatize('playing', pos="n"))

print(lemmatizer.lemmatize('playing', pos="a"))

print(lemmatizer.lemmatize('playing', pos="r"))

# play
#
# playing
#
# playing
#
# playing

# Stemming works on words without knowing its context
# and that’s why stemming has lower accuracy and faster than lemmatization.