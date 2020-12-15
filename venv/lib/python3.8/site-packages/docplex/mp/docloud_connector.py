# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

# pylint:skip-file
# gendoc: ignore

import json
import os

try:
    # python 2
    from urlparse import urlparse
except ImportError:
    # python 3
    from urllib.parse import urlparse


from datetime import datetime
from six import iteritems, string_types
from requests.exceptions import ConnectionError

import warnings

from docplex.util.status import JobSolveStatus

from docloud.job import JobClient, DOcloudInterruptedException, DOcloudNotFoundError
from docloud.status import JobExecutionStatus

from docplex.mp.utils import resolve_pattern, get_logger
from docplex.mp.utils import CyclicLoop

from docplex.mp.context import has_credentials


def key_as_string(key):
    """For keys, we don't want the key to appear in INFO log outputs.
    Instead, we display the first 4 chars and the last 4 chars.
    """
    return (key[:4] + "*******" + key[-4:]) if isinstance(key, string_types) else str(key)


def get_cplex_version(context):
    '''Submits a minimal job on DOCplexcloud and get the CPLEX version from
    logs
    Args:
        context: The context to use to connect
    Returns:
        A tuple (major, minor, micro), for instance (12, 7, 1)
    '''
    import re
    from docplex.mp.compat23 import StringIO
    from docplex.mp.model import Model
    logs = StringIO()
    model = Model()
    model.solve(url=context.solver.docloud.url,
                key=context.solver.docloud.key,
                log_output=logs)
    v = re.search('CPLEX version ([0-9][0-9])([0-9][0-9])([0-9][0-9])([0-9][0-9])', logs.getvalue())
    try:
        major = int(v.group(1))
        minor = int(v.group(2))
        micro = int(v.group(3))
        return (major, minor, micro)
    except IndexError:
        raise RuntimeError('Could not find CPLEX version in resulting logs')
    except:
        raise ValueError('Could not determine version from %s' % v.group(0))


class DOcloudConnectorException(Exception):
    def __init__(self, msg):
        Exception.__init__(self, msg)
        self.message = msg


class DOcloudEvent(object):
    """Used internally to manage events that must be processed in
    the main thread.

    The Progress and log monitor loop poll the service for progress and log
    information, then queue DOcloudEvents on an internal queue. That queue
    is processed by a loop in the main thread, making sure that logs are
    printed in the main thread and progress listeners are called in the
    main thread too.
    """
    def __init__(self, event_type, data):
        """The type if a string. Can be "log" or "progress"
        """
        self.type = event_type
        self.data = data


class DOcloudConnector(object):

    def __init__(self, docloud_context, log_output=None):
        """ Starts a connector which URL and authorization are stored in the
        specified context, along with other connection parameters

        The `docloud_context` refers to the context.solver.docloud node of a
        context.

        Args:
            log_output: The log output stream
        """
        if docloud_context is None or not has_credentials(docloud_context):
            raise DOcloudConnectorException("Please provide DOcplexcloud credentials")

        # store this for future usage
        self.docloud_context = docloud_context

        url = docloud_context.url
        auth = docloud_context.key

        self.logger = get_logger('DOcloudConnector', self.docloud_context.verbose)

        self.logger.info("DOcplexcloud connection using url = " + str(url) + " api_key = " + key_as_string(auth))
        self.logger.info("DOcplexcloud SSL verification = " + str(docloud_context.verify))

        self.logger.info("   waittime = " + str(docloud_context.waittime))
        self.logger.info("   timeout = " + str(docloud_context.timeout))

        self.jobInfo = None
        self.run_deterministic = docloud_context.run_deterministic
        self.log_output = log_output

        self.timed_out = False
        self.results = {}

    def log(self, msg, *args):
        if self.docloud_context.verbose:
            log_msg = "* {0}".format(resolve_pattern(msg, args))
            self.logger.info(log_msg)
            if self.log_output is not None:
                self.log_output.write(log_msg)

    def _as_string(self, content):
        resp_content_as_string = content
        if not isinstance(resp_content_as_string, str):
            resp_content_as_string = content.decode('utf-8')
        return resp_content_as_string

    def submit_model_data(self,
                          attachments=None,
                          gzip=False,
                          info_callback=None,
                          info_to_monitor=None):
        """Submits a job to the cloud service.

        Args:
            attachments: A list of attachments. Each attachement is a dict with
                the following keys:
                   - 'name' : the name of the attachment
                   - 'data' : the data for the attachment
            gzip: If ``True``, data is gzipped before sent over the network
            info_callback: A call back to be called when some info are available.
                That callback takes one parameter that is a dict containing
                the info as they are available.
            info_to_monitor: A set of information to monitor with info_callback.
                Currently, can be ``jobid`` and ``progress``.
        """
        self.__vars = None
        self.timed_out = False
        self.results.clear()

        if not info_to_monitor:
            info_to_monitor = {}

        # check that url is valid
        parts = urlparse(self.docloud_context.url)
        if not parts.scheme:
            raise DOcloudConnectorException("Malformed URL: '%s': No schema supplied." % self.docloud_context.url)

        proxies = self.docloud_context.proxies
        try:
            client = JobClient(self.docloud_context.url,
                               self.docloud_context.key,
                               proxies=proxies)
        except TypeError:
            # docloud client <= 1.0.172 do not have the proxes
            warnings.warn("Using a docloud client that do not support warnings in init()",
                          UserWarning)
            client = JobClient(self.docloud_context.url,
                               self.docloud_context.key)
        self.log("client created")
        if proxies:
            self.log("proxies = %s" % proxies)

        # prepare client
        if self.docloud_context.log_requests:
            client.rest_callback = \
                lambda m, u, *a, **kw: self._rest_callback(m, u, *a, **kw)
        client.verify = self.docloud_context.verify
        client.timeout = self.docloud_context.get('timeout', None)

        try:
            try:
                # Extract the list of attachment names
                att_names = [a['name'] for a in attachments]

                # create job
                jobid = client.create_job(attachments=att_names,
                                          parameters=self.docloud_context.job_parameters)
                self.log("job creation submitted, id is: {0!s}".format(jobid))
                if info_callback and 'jobid' in info_to_monitor:
                    info_callback({'jobid': jobid})
            except ConnectionError as c_e:
                raise DOcloudConnectorException("Cannot connect to {0}, error: {1}".format(self.docloud_context.url, str(c_e)))

            try:
                # now upload data
                for a in attachments:
                    pos = 0
                    if 'data' in a:
                        att_data = {'data': a['data']}
                    elif 'file' in a:
                        att_data = {'file': a['file']}
                        pos = a['file'].tell()
                    elif 'filename' in a:
                        att_data = {'filename': a['filename']}

                    client.upload_job_attachment(jobid,
                                                 attid=a['name'],
                                                 **att_data)
                    self.log("Attachment: %s has been uploaded" % a['name'])
                    if self.docloud_context.debug_dump_dir:
                        target_dir = self.docloud_context.debug_dump_dir
                        if not os.path.exists(target_dir):
                            os.makedirs(target_dir)
                        self.log("Dumping input attachment %s to dir %s" % (a['name'], target_dir))
                        with open(os.path.join(target_dir, a['name']), "wb") as f:
                            if 'data' in 'a':
                                if isinstance(a['data'], bytes):
                                    f.write(a['data'])
                                else:
                                    f.write(a['data'].encode('utf-8'))
                            else:
                                a['file'].seek(pos)
                                f.write(a['file'])
                # execute job
                client.execute_job(jobid)
                self.log("DOcplexcloud execute submitted has been started")
                # get job execution status until it's processed or failed
                timedout = False
                try:
                    self._executionStatus = self.wait_for_completion(client,
                                                                     jobid,
                                                                     info_callback=info_callback,
                                                                     info_to_monitor=info_to_monitor)
                except DOcloudInterruptedException:
                    timedout = True
                self.log("docloud execution has finished")
                # get job status. Do this before any time out handling
                self.jobInfo = client.get_job(jobid)

                if self.docloud_context.fire_last_progress and info_callback:
                    progress_data = self.map_job_info_to_progress_data(self.jobInfo)
                    info_callback({'progress': progress_data})

                if timedout:
                    self.timed_out = True
                    self.log("Solve timed out after {waittime} sec".format(waittime=self.docloud_context.waittime))
                    return
                # get solution => download all attachments
                try:
                    for a in client.get_job_attachments(jobid):
                        if a['type'] == 'OUTPUT_ATTACHMENT':
                            name = a['name']
                            self.log("Downloading attachment '%s'" % name)
                            attachment_as_string = self._as_string(client.download_job_attachment(jobid,
                                                                                                  attid=name))
                            self.results[name] = attachment_as_string
                            if self.docloud_context.debug_dump_dir:
                                target_dir = self.docloud_context.debug_dump_dir
                                if not os.path.exists(target_dir):
                                    os.makedirs(target_dir)
                                self.log("Dumping attachment %s to dir %s" % (name, target_dir))
                                with open(os.path.join(target_dir, name), "wb") as f:
                                    f.write(attachment_as_string.encode('utf-8'))
                except DOcloudNotFoundError:
                    self.log("no solution in attachment")
                self.log("docloud results have been received")
                # on_solve_finished_cb
                if self.docloud_context.on_solve_finished_cb:
                    self.docloud_context.on_solve_finished_cb(jobid=jobid,
                                                              client=client,
                                                              connector=self)
                return
            finally:
                if self.docloud_context.delete_job:
                    deleted = client.delete_job(jobid)
                    self.log("delete status for job: {0!s} = {1!s}".format(jobid, deleted))

        finally:
            client.close()

    def wait_for_completion(self, client, jobid,
                            info_callback=None, info_to_monitor=None):
        def status_poll(loop):
            """Callback to check execution status and stop the loop as soon
            as the job is finished. The loop is stopped as soon as the status
            is for a finished job,
            """
            status = loop.client.get_execution_status(loop.jobid)
            if JobExecutionStatus.isEnded(status):
                loop.status = status
                loop.stop()

        def waittime_timeout(loop):
            """Callback to stop completly the loop once the max waittime
            has been hit.
            """
            loop.stop()
            loop.timed_out = True

        def download_logs(loop, using_threads, log_output):
            """Function/Callback to download logs.

            We will use that function in threads (log_output will be None),
            were we want log events to be put in the loop's event_queue,
            and also after the loop has stopped(), to dump any log items
            remaining on the server (in this case, log_output will
            be the stream to write logs to)

            Args:
                loop: the loop
                using_threads: if true, we are using threads and should queue
                    events so that they are processed in the main thread
                log_output: the log_output
            """
            logs = loop.client.get_log_items(loop.jobid, loop.last_seqid, True)
            for log in logs:
                loop.last_seqid = log['seqid'] + 1
                for r in log['records']:
                    level = r['level'][:4]
                    date = r['date']
                    message = r['message'].rstrip()
                    d = datetime.utcfromtimestamp(int(float(date) * 0.001))
                    m = "[{date}Z, {level}] {message}\n".format(date=d.isoformat(),
                                                                level=level,
                                                                message=message
                                                                )
                    if using_threads:
                        loop.event_queue.put(DOcloudEvent("log", m))
                    else:
                        log_output.write(m)

        def progress_poll(loop, using_threads, info_callback, info_to_monitor):
            """Function/Callback to poll and download progress.

            That function polls the service for progress, then queue
            progress events in the main loop.

            Args:
                loop: the loop
                using_threads: if true, we are using threads and should queue
                    events so that they are processed in the main thread
                info_callback: the info callback
                info_to_monitor: what info does the callback want to monitor
            """
            if 'progress' in info_to_monitor and info_callback:
                logger = self.docloud_context.verbose_progress_logger
                if logger:
                    logger.info("polling progress")
                info = loop.client.get_job(loop.jobid)
                if logger:
                    logger.info("job info: %s" % json.dumps(info, indent=3))
                if 'details' in info:  # there are some info available
                    if logger:
                        logger.info("generating progress event using_threads = %s" % using_threads)
                    progress_data = self.map_job_info_to_progress_data(info)
                    if using_threads:
                        loop.event_queue.put(DOcloudEvent("progress", progress_data))
                    else:
                        info_callback({'progress': progress_data})

        class JobMonitor(CyclicLoop):
            """A cyclic loop with some encapsuled data"""
            def __init__(self, client, jobid, log_output):
                super(JobMonitor, self).__init__()
                self.client = client
                self.jobid = jobid
                self.log_output = log_output

                self.status = None
                self.timed_out = False
                self.last_seqid = 0

        if not info_to_monitor:
            info_to_monitor = {}

        # interval to check job status
        status_poll_interval = client.nice
        log_poll_interval = self.docloud_context.get('log_poll_interval')
        if log_poll_interval is None:
            log_poll_interval = client.nice * 3
        progress_poll_interval = self.docloud_context.get('progress_poll_interval')
        if progress_poll_interval is None:
            progress_poll_interval = client.nice * 3

        # The cyclic loop
        loop = JobMonitor(client, jobid, self.log_output)
        using_threads = False

        # configure status log event
        loop.enter(status_poll_interval, 1, status_poll, (loop, ))
        # configure log poll event
        if self.log_output:
            self.log("Polling logs every %s sec" % log_poll_interval)
            loop.enter(log_poll_interval, 1, download_logs,
                       (loop, using_threads, self.log_output))
        # configure progress poll event
        if 'progress' in info_to_monitor:
            self.log("Polling progress every %s sec" % progress_poll_interval)
            loop.enter(progress_poll_interval, 1,
                       progress_poll,
                       (loop, using_threads, info_callback, info_to_monitor))

        # If there's a waittime, configure an event to stop the loop after
        # ``waittime``
        if self.docloud_context.waittime:
            loop.enter(self.docloud_context.waittime, 1, waittime_timeout, (loop, ))
            self.log("waiting for job completion with a wait time of {waittime} sec".format(waittime=self.docloud_context.waittime))
        else:
            self.log("waiting for job completion with no wait time")

        # we want the dumps of the log to happen in the main thread
        # we also want the progress listener events to come in the main thread
        # this function is guaranteed to run in the main thread by the loop
        def main_thread_worker(event):
            if event.type == "log":
                loop.log_output.write(event.data)
            elif event.type == "progress":
                if info_callback:
                    info_callback({'progress': event.data})

        kwargs = {}
        if using_threads:
            kwargs['mt_worker'] = main_thread_worker
        loop.start(**kwargs)

        if self.log_output:
            # this will download the log items that were not downloaded in
            # the loop. Using using_threads=False to force a dump of the log
            # output
            download_logs(loop, False, self.log_output)

        if loop.timed_out:
            self.log("Job Timed out")
            raise DOcloudInterruptedException("Timeout after {0}".format(self.docloud_context.waittime), jobid=jobid)

        return loop.status

    def get_cplex_details(self):
        if self.jobInfo:
            return self.jobInfo.get("details")

    def get_solve_status(self):
        if 'solveStatus' in self.jobInfo:
            return JobSolveStatus[self.jobInfo['solveStatus']]
        else:
            return None

    def _rest_callback(self, method, url, *args, **kwargs):
        """The callback called by the DOcplexcloud client to log REST operations
        """
        self.logger.info("{0} {1}".format(method, url))
        if len(args) > 0:
            self.logger.info("   Additionnal args : {0}".format(','.join(args)))
        for k, v in iteritems(kwargs):
            self.logger.info("   {0}: {1}".format(k, v))

    def map_job_info_to_progress_data(self, info):
        """ Map job info as downloaded from the cplex cloud worker to
        docplex.mp.progress.ProgressData

        Args:
            info: The info as a dict
        Returns:
            A ProgressData
        """
        from docplex.mp.progress import ProgressData
        has_incumbent = current_objective = best_bound = mip_gap = current_nb_iterations = current_nb_nodes = remaining_nb_nodes = 0
        time = 0
        details = info.get('details')
        if details:
            current_objective = float(details.get('PROGRESS_CURRENT_OBJECTIVE', 0))
            best_bound = float(details.get('PROGRESS_BEST_OBJECTIVE', 0))
            mip_gap = None
            if 'PROGRESS_CURRENT_OBJECTIVE' in details and 'PROGRESS_BEST_OBJECTIVE' in details:
                if current_objective > 0:
                    mip_gap = abs(current_objective - best_bound) / current_objective
            current_nb_nodes = int(details.get('cplex.nodes.processed', 0))
            remaining_nb_nodes = int(details.get('cplex.nodes.left', 0))
            # assume that there's an incubent if ther's a gap
            has_incumbent = 'PROGRESS_GAP' in details
        if 'startedAt' in info and 'updatedAt' in info:
            time = ((info.get('updatedAt')) - int(info.get('startedAt'))) / 1000
        pg = ProgressData(0, has_incumbent, current_objective, best_bound, mip_gap,
                          0, current_nb_nodes, remaining_nb_nodes,
                          time, 0)
        return pg
