# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis
#         with help from previous version written by Viu LONG KONG

"""
This module implements a DOcplexcloud client allowing to submit a CPO model for solving on DOcplexcloud.
"""

import time, json
import requests

from docplex.cp.utils import CpoException, is_symbol_char, Context


###############################################################################
##  Constants
###############################################################################

# Job execution statuses
ALL_JOB_STATUSES = ('CREATED', 'NOT_STARTED', 'RUNNING', 'INTERRUPTING', 'INTERRUPTED', 'FAILED', 'PROCESSED')

# Default polling parameters
_DEFAULT_POLLING = Context(min=1, max=3, incr=0.2)

# Unknown job id
_UNKNOWN_JOB_ID = "Unknown"


###############################################################################
##  Public classes
###############################################################################

class DocloudException(CpoException):
    """ The base class for exceptions raised by the DOcplexcloud client
    """
    def __init__(self, msg):
        """ Create a new exception
        Args:
            msg: Error message
        """
        super(DocloudException, self).__init__(msg)


class JobClient(object):
    """ A client to create, submit and monitor one job on DOcplexcloud. """
    __slots__ = ('ctx',      # DOcplexcloud context parameters
                 'jobid',    # Id of the job handled by this client
                 'headers',  # Default headers
                )

    def __init__(self, ctx):
        """ Initialize a new job client.
        Args:
            ctx: DOcplexcloud context
        """
        self.ctx = ctx
        self.jobid = _UNKNOWN_JOB_ID
        # Build default headers
        self.headers = {'X-IBM-Client-Id': ctx.key, 'Content-Type': 'application/json'}
        if ctx.always_close_connection:
            self.headers['Connection'] = 'close'
        # Disable warnings if verify sll is false
        if not ctx.verify_ssl:
            requests.packages.urllib3.disable_warnings()


    def create_job(self, attachments, parameters={}):
        """ Create a new model solving job. JobId is stored in this object.
        Args:
            attachments: List of attachments. Each attachment is a tuple (name, content)
            parameters:  Optional map of parameters
        """
        # Create job
        args = {'attachments': [{'name': name} for name, content in attachments],
                'parameters': parameters}
        rsp = self._request('post', self.ctx.url + "/jobs", [201], data=json.dumps(args))
        self.jobid = rsp.headers['location'].rsplit("/", 1)[1]
        self.ctx.log(2, "Job created: ", self.jobid)
        # Upload attachments
        hdrs = self.headers.copy()
        hdrs['Content-Type'] = 'application/octet-stream'
        for name, content in attachments:
            self._request('put', self.ctx.url + "/jobs/" + self.jobid + "/attachments/" + name + "/blob", [204], data=content, headers=hdrs)
            self.ctx.log(2, "Attachment '", name, "' uploaded (", len(content), " bytes)")


    def create_cpo_job(self, name, cpodata, cmd='Solve'):
        """ Create a new model solving job. JobId is stored in this object.
        Args:
            name:    Job name
            cpodata: Model to be solved in CPO format, already encoded in UTF8
            cmd:     Command name
        """
        # Create job
        self.create_job([(normalize_job_name(name) + ".cpo", cpodata)], {'oaas.cpo.command': cmd})


    def execute_job(self):
        """ Request job execution """
        self._request('post', self.ctx.url + "/jobs/" + self.jobid + "/execute", [204])
        self.ctx.log(2, "Model solving requested")


    def get_info(self):
        """ Get the job information
        Returns:
            Job information as JSON document
        """
        rsp = self._request('get', self.ctx.url + "/jobs/" + self.jobid, [200])
        return rsp.json()


    def get_status(self):
        """ Get the job status
        Returns:
            Job execution status string, in 'INTERRUPTED', 'FAILED', 'PROCESSED', etc
        """
        rsp = self._request('get', self.ctx.url + "/jobs/" + self.jobid, [200])
        status = rsp.json()["executionStatus"]
        self.ctx.log(3, "Job status is ", status)
        return status


    def wait_job_termination(self, maxwait=0, lognotif=None):
        """ Wait for termination of the job
        Args:
            maxwait:  Maximum wait time in seconds, 0 for infinite. Default is zero.
            lognotif: Function allowing to notify log records when available. Default is none (no log).
        Returns:
            Status string
        Raises:
            DocloudException if wait timeout reached
        """
        self.ctx.log(2, "Waiting model solving termination.")

        # Retrieve polling parameters
        plprms = self.ctx.get_attribute('polling', _DEFAULT_POLLING)
        plmin  = plprms.get_attribute('min',  _DEFAULT_POLLING.min)
        plmax  = plprms.get_attribute('max',  _DEFAULT_POLLING.max)
        plincr = plprms.get_attribute('incr', _DEFAULT_POLLING.incr)

        # Initialize continuous log polling
        logseqid = 0

        # Initialize waiting loop
        etime = time.time() + maxwait
        wdelay = plmin
        terminated = False
        status = 'UNKNOWN'
        while not terminated:
            status = self.get_status()
            terminated = status in ('INTERRUPTED', 'FAILED', 'PROCESSED')
            if not terminated:
                if (maxwait > 0) and (time.time() > etime):
                    self._raise_exception("Timeout of " + str(maxwait) + " sec elapsed before job termination.")
                # Wait delay and increment it if possible
                time.sleep(wdelay)
                if wdelay < plmax:
                    wdelay += plincr
            # Get log if needed
            if lognotif is not None:
                log = self.get_log_items(logseqid)
                if log:
                    for li in log:
                        lognotif(li['records'])
                    logseqid = log[-1]['seqid'] + 1
        self.ctx.log(2, "Solve terminated. Status is ", status)
        return status


    def get_attachment(self, aname):
        """ Get a job attachment
        Args:
            aname: Attachment name
        Returns:
            Attachment content as a string
        """
        rsp = self._request('get', self.ctx.url + "/jobs/" + self.jobid + "/attachments/" + aname + "/blob", [200])
        return rsp.content


    def get_log_blob(self):
        """ Get the whole job execution log as a string.
        Returns:
            The whole job log as a string, to be decoded in UTF8.
        """
        rsp = self._request('get', self.ctx.url + "/jobs/" + self.jobid + "/log/blob", [200])
        self.ctx.log(2, "Log retrieved")
        return rsp.content


    def get_log_items(self, start):
        """ Get the job execution log.
        Args:
            start: Log start index
        Returns:
            Log items as JSON document
        """
        # Build headers
        rsp = self._request('get', self.ctx.url + "/jobs/" + self.jobid + "/log/items?start=" + str(start) + "&continuous=true", [200])
        self.ctx.log(2, "Log items retrieved")
        return rsp.json()


    def abort_job(self):
        """ Abort the job handled by this client. """
        self._request('delete', self.ctx.url + "/jobs/" + self.jobid + "/execute", [204])


    def delete_job(self):
        """ Delete the job handled by this client.
        Returns:
            True if the job deletion was successful
        """
        rsp = self._request('delete', self.ctx.url + "/jobs/" + self.jobid, [200])
        return rsp.json().get('status', None) == 'DELETED'


    def clean_job(self):
        """ Clean a job (abort and delete)
        This method should be called to cancel and clean a job in case of error
        No exception is thrown if an error occur
        """
        if self.jobid != _UNKNOWN_JOB_ID:
            try:
                self.abort_job()
            except Exception:
                pass
            try:
                self.delete_job()
            except Exception:
                pass
        self.ctx.log(2, "Clean job " + str(self.jobid) + " terminated")


    def get_all_jobs(self):
        """ Return all jobs for this account.
        Returns:
            List of job information.
            Each Job information is a dict containing the job information, as returned by get_info() for a given jobid.
        """
        rsp = self._request('get', self.ctx.url + "/jobs", [200])
        return rsp.json()


    def map_to_job(self, jobid):
        """ Set this job client for a given jobid
        Args:
            jobid: Job id to address
        """
        self.jobid = jobid


    def clean_all_jobs(self):
        """ Clear all jobs attached to this user.
        Returns:
            List of job information of the deleted jobs.
        """
        ljobs = self.get_all_jobs()
        for job in ljobs:
            self.map_to_job(job['_id'])
            self.clean_job()
        return ljobs


    def _request(self, mth, url, astc, **kwargs):
        """ Send a request to DOcplexcloud with default headers and check response.
        Args:
            mth:     HTTP method name
            url:     Target url
            astc:    List of expected status codes
            kwargs:  Arguments to pass to request
        Returns:
            Request response
        """
        # Build requests arguments
        ctx = self.ctx
        kwargs.setdefault('headers',         self.headers)
        kwargs.setdefault('verify',          ctx.verify_ssl)
        kwargs.setdefault('timeout',         ctx.request_timeout)
        kwargs.setdefault('allow_redirects', True)
        proxies = ctx.proxies
        if proxies:
            kwargs.setdefault('proxies', proxies)
        # print("Arguments: " + str(kwargs))

        # Send request
        ctx.log(4, "Send request ", mth, " ", url)
        try:
            rsp = requests.request(mth, url, **kwargs)
        except Exception as e:
            self._raise_exception("Rest request error :" + str(e))

        # Check response
        sc = rsp.status_code
        ctx.log(4, "Response code: ", sc)
        if not (sc in astc):
            # Get message if any
            try:
                j = rsp.json()
                message = j["message"]
            except ValueError:
                message = str(rsp)
            if sc == 403:
                message = "Access forbidden: " + message
            elif sc == 404:
                message = "Page not found: " + message
            self._raise_exception(message)
        # Return
        return rsp


    def _raise_exception(self, msg):
        """ Raise a DocloudException
        Args:
            msg:  Exception message
        """
        raise DocloudException("Job " + str(self.jobid) + " error: " + msg)


def normalize_job_name(name):
    """ Normalize a job name by replacing unallowed characters by '_'

    If the name is empty, a name is generated with current time.

    Args:
        name:  Name to normalize
    Returns:
        Normalized name
    """
    name = str(name).strip()
    if name == "":
        name = "Job_" + str(time.time())
    else:
        name = ''.join([c if is_symbol_char(c) else '_' for c in name])
        if not name[0].isalpha():
            name = "Job_" + name
    return name


