import os
import tempfile
import heapq
import sys
import pickle
import glob


class heapnode:
    """ Heapnode of a Heap (MinHeap Here)
       @params
            item        The actual value to be stored in heap
            fileHandler The filehandler of the file that stores the number
    """
    def __init__(self, item, fileHandler):
        self.item = item
        self.fileHandler = fileHandler


class externamMergeSort:
    """ Splits the large file into small files ,sort the small files
        and uses python heapq module to merge the different small sorted files.
        Each sorted file is loaded as a generator, hence won't load entire data
        into memory
        @params
           sortedTempFileHandlerList    List of all filehandlers to all temp
           files formed by splitting large files
    """
    def __init__(self):
        self.sortedTempFileHandlerList = []
        self.getCurrentDir()

    def getCurrentDir(self):
        self.cwd = os.getcwd()

    """ Iterates the sortedCompleteData Generator """
    def iterateSortedData(self, sortedCompleteData):
        for no in sortedCompleteData:
            print(no)

    """ HighLevel Pythonic way to sort all numbers in the list of files
        pointed by Filehandlers of sortedTempFileHandlerList
    """
    def mergeSortedtempFiles(self):
        # mergedNo is a generator storing all the sorted number
        # in ((1,4,6),(3,7,8)) format.
        # It doesn't stores in memory and do lazy allocation
        mergedNo = (map(int, tempFileHandler) for tempFileHandler in self.sortedTempFileHandlerList)
        # uses heapqmodule that takes a list of sorted iterators
        # and sort it and generates a sorted iterator
        sortedCompleteData = heapq.merge(*mergedNo)
        return sortedCompleteData

    """ min heapify function """
    def heapify(self, arr, i, n):
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n and arr[left].item[0] < arr[i].item[0]:
            smallest = left
        else:
            smallest = i

        if right < n and arr[right].item[0] < arr[smallest].item[0]:
            smallest = right

        if i != smallest:
            (arr[i], arr[smallest]) = (arr[smallest], arr[i])
            self.heapify(arr, smallest, n)

    """ construct heap """

    def construct_heap(self, arr):
        l = len(arr) - 1
        mid = int(l / 2)
        while mid >= 0:
            self.heapify(arr, mid, l)
            mid -= 1

    """ low level implementation to merge k sorted small file to a larger file.
        Move first element of all files to a min heap.
        Heap has now the smallest element.
        Moves that element from heap to file.
        Get the filehandler of that element.
        Read the next element using the same filehandler.
        If next file element is empty, mark it as INT_MAX. Moves it to heap.
        Again Heapify.
        Continue until all elements of heap is INT_MAX or all smaller files have read.
    """

    def mergeSortedtempFiles_low_level(self, outfilename):
        L = []
        sorted_output = []
        for tempFileHandler in self.sortedTempFileHandlerList:
            line = tempFileHandler.readline().decode().strip()
            if line:
                lst = line.split(" ")
                item = tuple([int(lst[0]), int(lst[1])])
                L.append(heapnode(item, tempFileHandler))

        self.construct_heap(L)
        while True:
            minN = L[0]
            if minN.item[0] == sys.maxsize:
                break
            sorted_output.append(minN.item)
            fileHandler = minN.fileHandler
            line = fileHandler.readline().decode().strip()
            if line:
                lst = line.split(" ")
                item = tuple([int(lst[0]), int(lst[1])])
            else:
                item = tuple([sys.maxsize, 0])
            L[0] = heapnode(item, fileHandler)
            self.heapify(L, 0, len(L))
        with open(outfilename, 'wb') as out_f:
            for line in sorted_output:
                out_f.write((f'{line[0]} {line[1]}\n').encode(encoding='UTF-8'))
        return 'Done'

    """ function to split a large files into smaller chunks,
        sort them and store it to temp files on disk
    """
    def splitFiles(self, largeFileName, smallFileSize):
        largeFileHandler = open(largeFileName)
        tempBuffer = []
        size = 0
        while True:
            number = largeFileHandler.readline().strip()
            if not number and len(tempBuffer) > 0:
                tempBuffer = sorted(tempBuffer, key=lambda no: int(no.split(" ")[0]))
                tmp = tempfile.NamedTemporaryFile(dir=self.cwd, delete=True)
                for line in tempBuffer:
                    tmp.write((line + '\n').encode(encoding='UTF-8'))
                tmp.seek(0)
                self.sortedTempFileHandlerList.append(tmp)
                break
            tempBuffer.append(number)
            size += 1
            if size % smallFileSize == 0:
                tempBuffer = sorted(tempBuffer, key=lambda no: int(no.split(" ")[0]))
                tmp = tempfile.NamedTemporaryFile(dir=self.cwd, delete=True)
                for line in tempBuffer:
                    tmp.write((line + '\n').encode(encoding='UTF-8'))
                tmp.seek(0)
                self.sortedTempFileHandlerList.append(tmp)
                tempBuffer = []


if __name__ == '__main__':
    largeFileName = sys.argv[1]
    smallFileSize = 100000000
    obj = externamMergeSort()
    obj.splitFiles(largeFileName, smallFileSize)
    obj.mergeSortedtempFiles_low_level()
